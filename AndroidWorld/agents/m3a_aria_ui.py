# Copyright 2024 The Aria-UI Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A Multimodal Autonomous Agent for Android (Aria-UI under M3A)."""

import time
from android_world.agents import agent_utils
from android_world.agents import base_agent
from android_world.agents import infer
from android_world.agents import m3a_utils
from android_world.env import interface
from android_world.env import json_action
from android_world.env import representation_utils
from android_world.agents import aria_ui_utils
import ast
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


PROMPT_PREFIX = """
You are an agent who can operate an Android phone on behalf of a user. Based on the user's goal/request, you may:

- Answer back if the request/goal is a question (or a chat message), like user asks "What is my schedule for today?".  
- Complete some tasks described in the requests/goals by performing actions (step by step) on the phone.

When given a user request, you will try to complete it step by step. At each step, you will be given the current screenshot and a history of what you have done (in text). Based on these pieces of information and the goal, you must choose to perform one of the actions in the following list (action description followed by the JSON format) by outputting the action in the correct JSON format:

- If you think the task has been completed, finish the task by using the status action with complete as goal_status:  
  {{
    "action_type": "status", 
    "goal_status": "complete"
  }}

- If you think the task is not feasible (including cases like you don't have enough information or cannot perform some necessary actions), finish by using the `status` action with infeasible as goal_status:  
  {{
    "action_type": "status", 
    "goal_status": "infeasible"
  }}

- Answer user's question:  
  {{
    "action_type": "answer", 
    "text": "answer_text"
  }}

- Click/tap on an element on the screen. Please describe the element you want to click using natural language:  
  {{
    "action_type": "click", 
    "instruction": the step-wise instruction in short,
    "target": target_element_description
  }}

- Long press on an element on the screen, similar to the click action above, use the semantic description to indicate the element:  
  {{
    "action_type": "long_press", 
    "instruction": the step-wise instruction in short,
    "target": target_element_description
  }}

- Type text into a text field (this action contains clicking the text field, typing in the text, and pressing Enter, so no need to click on the target field to start). Use the semantic description to indicate the target text field:  
  {{
    "action_type": "input_text", 
    "text": text_input, 
    "instruction": the step-wise instruction in short,
    "target": target_element_description
  }}

- Press the Enter key:  
  {{
    "action_type": "keyboard_enter"
  }}

- Navigate to the home screen:  
  {{
    "action_type": "navigate_home"
  }}

- Navigate back:  
  {{
    "action_type": "navigate_back"
  }}

- Scroll the screen or a scrollable UI element in one of the four directions, use the same semantic description as above if you want to scroll a specific UI element, leave it empty when scrolling the whole screen:  
  {{
    "action_type": "scroll", 
    "direction": "up/down/left/right", 
    "instruction": the step-wise instruction in short,
    "element": optional_target_element_description
  }}

- Open an app (nothing will happen if the app is not installed):  
  {{
    "action_type": "open_app", 
    "app_name": name
  }}

- Wait for the screen to update:  
  {{
    "action_type": "wait"
  }}
"""

GUIDANCE = """
Here are some useful guidelines you need to follow:

General:
- Usually there will be multiple ways to complete a task, pick the easiest one. Also when something does not work as expected (due to various reasons), sometimes a simple retry can solve the problem, but if it doesn’t (you can see that from the history), SWITCH to other solutions.
- Sometimes you may need to navigate the phone to gather information needed to complete the task, for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app (using the ‘open_app‘ action), look up information there, answer user’s question (using the ‘answer‘ action) and finish (using the ‘status‘ action with complete as goal_status).
- For requests that are questions (or chat messages), remember to use the ‘answer‘ action to reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient (unless the goal is something like "show me ...").
- If the desired state is already achieved (e.g., enabling Wi-Fi when it’s already on), you can just complete the task.

Action Related:
- Use the ‘open_app‘ action whenever you want to open an app (nothing will happen if the app is not installed), do not use the app drawer to open an app unless all other ways have failed.
- Use the ‘input_text‘ action whenever you want to type something (including passwords) instead of clicking characters on the keyboard one by one. Sometimes there is some default text in the text field you want to type in, remember to delete them before typing.
- For ‘click‘, ‘long_press‘ and ‘input_text‘, the target_element.description parameter you choose must be based on a VISIBLE element in the screenshot.
- Consider exploring the screen by using the ‘scroll‘ action with different directions to reveal additional content.
- The direction parameter for the ‘scroll‘ action can be confusing sometimes as it’s opposite to swipe, for example, to view content at the bottom, the ‘scroll‘ direction should be set to "down". It has been observed that you have difficulties in choosing the correct direction, so if one does not work, try the opposite as well.

Text Related Operations:
- Normally to select certain text on the screen: (i) Enter text selection mode by long pressing the area where the text is, then some of the words near the long press point will be selected (highlighted with two pointers indicating the range) and usually a text selection bar will also appear with options like "copy", "paste", "select all", etc. (ii) Select the exact text you need. Usually the text selected from the previous step is NOT the one you want, you need to adjust the range by dragging the two pointers. If you want to select all text in the text field, simply click the "select all" button in the bar.
- At this point, you don’t have the ability to drag something around the screen, so in general you can not select arbitrary text.
- To delete some text: the most traditional way is to place the cursor at the right place and use the backspace button in the keyboard to delete the characters one by one (can long press the backspace to accelerate if there are many to delete). Another approach is to first select the text you want to delete, then click the backspace button in the keyboard.
- To copy some text: first select the exact text you want to copy, which usually also brings up the text selection bar, then click the `copy` button in bar.
- To paste text into a text box, first long press the text box, then usually the text selection bar will appear with a `paste` button in it.
- When typing into a text field, sometimes an auto-complete dropdown list will appear. This usually indicating this is a enum field and you should try to select the best match by clicking the corresponding one in the list.
"""


ACTION_SELECTION_PROMPT_TEMPLATE = (
    PROMPT_PREFIX + "\nThe current user goal/request is: {goal}\n\n"
    "Here is a history of what you have done so far:\n{history}\n\n"
    "The current screenshot is also given to you.\n"
    + GUIDANCE
    + "{additional_guidelines}"
    + "\nNow output an action from the above list in the correct JSON format,"
    " following the reason why you do that. Your answer should look like:\n"
    'Reason: ...\nAction: {{"action_type":...}}\n\n'
    "Your Answer:\n"
)


SUMMARY_PROMPT_TEMPLATE = (
    PROMPT_PREFIX + "\nThe (overall) user goal/request is: {goal}\n"
    "Now I want you to summerize the latest step.\n"
    "You will be given the screenshot before you performed the action (which"
    ' has a text label "before" on the bottom right), the action you chose'
    " (together with the reason) and the screenshot after the action was"
    ' performed (which has a text label "after" on the bottom right).\n'
    "Also here is the list of detailed information for some UI elements"
    " in the before screenshot:\n{before_elements}\n"
    "Here is the list for the after screenshot:\n{after_elements}\n"
    "This is the action you picked: {action}\n"
    "Based on the reason: {reason}\n\n"
    "By comparing the two screenshots (plus the UI element lists) and the"
    " action performed, give a brief summary of this step. This summary"
    " will be added to action history and used in future action selection,"
    " so try to include essential information you think that will be most"
    " useful for future action selections like what you"
    " intended to do, why, if it worked as expected, if not"
    " what might be the reason (be critical, the action/reason might be"
    " wrong), what should/should not be done next and so on. Some more"
    " rules/tips you should follow:\n"
    "- Keep it short (better less than 50 words) and in a single line\n"
    "- Some actions (like `answer`, `wait`) don't involve screen change,"
    " you can just assume they work as expected.\n"
    "- Given this summary will be added into action history, it can be used as"
    " memory to include information that needs to be remembered, or shared"
    " between different apps.\n\n"
    "Summary of this step: "
)


ARIA_UI_PROMPT_TEMPLATE = """The agent is performing the ultimate task: {ultimate_task}.
History of the agent's steps:\n{history_list}.
<image>Step {step_idx}. Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: {instruction}"""

ARIA_UI_PROMPT_TEMPLATE_MINIWOB = """<image>Given a GUI image, what are the relative (0-1000) pixel point coordinates for the element corresponding to the following instruction or description: {instruction}"""


def _generate_ui_element_description(
    ui_element: representation_utils.UIElement, index: int
) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + "}"


def _generate_ui_elements_description_list(
    ui_elements: list[representation_utils.UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: The height and width of the screen in pixels.

    Returns:
      Concise information for each UIElement.
    """
    tree_info = ""
    for index, ui_element in enumerate(ui_elements):
        if m3a_utils.validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + "\n"
    return tree_info


def _pvision_action_selection_prompt(
    goal: str,
    history: list[str],
    additional_guidelines: list[str] | None = None,
) -> str:
    """Generate the prompt for the action selection.

    Args:
      goal: The current goal.
      history: Summaries for previous steps.
      ui_elements: A list of descriptions for the UI elements.
      additional_guidelines: Task specific guidelines.

    Returns:
      The text prompt for action selection that will be sent to gpt4v.
    """
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    extra_guidelines = ""
    if additional_guidelines:
        extra_guidelines = "For The Current Task:\n"
        for guideline in additional_guidelines:
            extra_guidelines += f"- {guideline}\n"

    return ACTION_SELECTION_PROMPT_TEMPLATE.format(
        goal=goal,
        history=history,
        additional_guidelines=extra_guidelines,
    )


def _summarize_prompt(
    action: str,
    reason: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reason: The reason to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(
        goal=goal,
        before_elements=before_elements,
        after_elements=after_elements,
        action=action,
        reason=reason,
    )


def _extract_coords_from_response(response: str) -> tuple[int, int]:
    """Extract coordinate tuple from LLM response string.

    Args:
        response: String containing coordinates like "(892,925)" or "[892, 925]"

    Returns:
        Tuple of (x,y) coordinates as integers

    Raises:
        ValueError: If exactly 2 numbers are not found in the response
    """
    # Clean up the response string
    resp = response.replace("```", "").strip()

    # Extract numbers using regex
    import re

    numbers = re.findall(r"\d+", resp)
    if len(numbers) != 2:
        raise ValueError(
            f"Expected exactly 2 coordinates, found {len(numbers)} numbers in response: {response}"
        )

    return (int(numbers[0]), int(numbers[1]))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_fixed(5),
    retry=retry_if_exception_type(ValueError),
    reraise=True,
)
def call_grounding_llm(screenshot, goal, history, elem_description, elem_instruction):
    history_list = "\n".join(
        [f"\t{j+1}. " + step_info["summary"] for j, step_info in enumerate(history)]
    )
    """
    AndroidWorld
    """
    prompt = ARIA_UI_PROMPT_TEMPLATE.format(
        ultimate_task=goal,
        history_list=history_list,
        step_idx=len(history) + 1,
        instruction=f"description: {elem_description}; instruction: {elem_instruction}",
    )
    """
    MiniWob++
    """
    # prompt = ARIA_UI_PROMPT_TEMPLATE_MINIWOB.format(
    #     instruction=f"{elem_description}",
    # )

    response = aria_ui_utils.request_aria_ui(screenshot, prompt)

    coords = _extract_coords_from_response(response)
    return coords


class M3A(base_agent.EnvironmentInteractingAgent):
    """M3A which stands for Multimodal Autonomous Agent for Android."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm: infer.MultimodalLlmWrapper,
        name: str = "M3A",
        wait_after_action_seconds: float = 2.0,
    ):
        """Initializes a M3A Agent.

        Args:
          env: The environment.
          llm: The multimodal LLM wrapper.
          name: The agent name.
          wait_after_action_seconds: Seconds to wait for the screen to stablize
            after executing an action
        """
        super().__init__(env, name)
        self.llm = llm
        self.history = []
        self.additional_guidelines = None
        self.wait_after_action_seconds = wait_after_action_seconds

    def set_task_guidelines(self, task_guidelines: list[str]) -> None:
        self.additional_guidelines = task_guidelines

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        # Hide the coordinates on screen which might affect the vision model.
        self.env.hide_automation_ui()
        self.history = []

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "before_ui_elements": [],
            "after_screenshot_with_som": None,
            "action_prompt": None,
            "action_output": None,
            "action_output_json": None,
            "action_reason": None,
            "action_raw_response": None,
            "summary_prompt": None,
            "summary": None,
            "summary_raw_response": None,
        }
        print("----------step " + str(len(self.history) + 1))

        state = self.get_post_transition_state()
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary

        before_ui_elements = state.ui_elements
        step_data["before_ui_elements"] = before_ui_elements
        before_ui_elements_list = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        step_data["raw_screenshot"] = state.pixels.copy()
        before_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(before_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        action_prompt = _pvision_action_selection_prompt(
            goal,
            [
                "Step " + str(i + 1) + "- " + step_info["summary"]
                for i, step_info in enumerate(self.history)
            ],
            self.additional_guidelines,
        )
        step_data["action_prompt"] = action_prompt
        action_output, is_safe, raw_response = self.llm.predict_mm(
            action_prompt,
            [
                step_data["raw_screenshot"],
            ],
        )

        if is_safe == False:  # pylint: disable=singleton-comparison
            #  is_safe could be None
            action_output = f"""Reason: {m3a_utils.TRIGGER_SAFETY_CLASSIFIER}
Action: {{"action_type": "status", "goal_status": "infeasible"}}"""

        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")
        step_data["action_output"] = action_output
        step_data["action_raw_response"] = raw_response

        reason, action = m3a_utils.parse_reason_action_output(action_output)

        # If the output is not in the right format, add it to step summary which
        # will be passed to next step and return.
        if (not reason) or (not action):
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        print("Action: " + action)
        print("Reason: " + reason)
        step_data["action_reason"] = reason

        try:
            action_json = agent_utils.extract_json(action)
            converted_action = json_action.JSONAction(
                **action_json,
            )
            step_data["action_output_json"] = converted_action
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)

            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        action_index = converted_action.index
        elem_description = converted_action.target
        elem_instruction = converted_action.instruction

        num_ui_elements = len(before_ui_elements)
        if (
            converted_action.action_type
            in ["click", "long_press", "input_text", "scroll"]
            and elem_description is not None
        ):
            aria_ui_coords = call_grounding_llm(
                step_data["raw_screenshot"],
                goal,
                self.history,
                elem_description,
                elem_instruction,
            )
            physical_coords = aria_ui_utils.convert_coords_to_physical(
                aria_ui_coords,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )

            converted_action.x = physical_coords[0]
            converted_action.y = physical_coords[1]

            # Add mark to the target element.
            aria_ui_utils.add_ui_element_mark_coords(
                step_data["raw_screenshot"],
                aria_ui_coords,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )

        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            step_data["summary"] = "Agent thinks the request has been completed."
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                True,
                step_data,
            )

        if converted_action.action_type == "answer":
            print("Agent answered with: " + converted_action.text)

        try:
            self.env.execute_action(converted_action)
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to execute action.")
            print(str(e))
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        time.sleep(self.wait_after_action_seconds)

        state = self.env.get_state(wait_to_stabilize=False)
        logical_screen_size = self.env.logical_screen_size
        orientation = self.env.orientation
        physical_frame_boundary = self.env.physical_frame_boundary
        after_ui_elements = state.ui_elements
        after_ui_elements_list = _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        after_screenshot = state.pixels.copy()
        for index, ui_element in enumerate(after_ui_elements):
            if m3a_utils.validate_ui_element(ui_element, logical_screen_size):
                m3a_utils.add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        m3a_utils.add_screenshot_label(
            step_data["before_screenshot_with_som"], "before"
        )
        m3a_utils.add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            before_ui_elements_list,
            after_ui_elements_list,
        )
        summary, is_safe, raw_response = self.llm.predict_mm(
            summary_prompt,
            [
                before_screenshot,
                after_screenshot,
            ],
        )

        if is_safe == False:  # pylint: disable=singleton-comparison
            #  is_safe could be None
            summary = """Summary triggered LLM safety classifier."""

        if not raw_response:
            print(
                "Error calling LLM in summarization phase. This should not happen: "
                f"{summary}"
            )
            step_data["summary"] = (
                "Some error occurred calling LLM during summarization phase: %s"
                % summary
            )
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                False,
                step_data,
            )

        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        print("Summary: " + summary)
        step_data["summary_raw_response"] = raw_response

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            False,
            step_data,
        )
