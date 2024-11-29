import argparse
import base64
import io
import numpy as np
from PIL import Image

import re

from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html, overlay_som

# OTHER MSIC. UTILS


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--visual_effects",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html",
        type=str2bool,
        default=False,
        help="Use HTML in the agent's observation space.",
    )
    parser.add_argument(
        "--use_axtree",
        type=str2bool,
        default=True,
        help="Use AXTree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=False,
        help="Use screenshot in the agent's observation space.",
    )

    return parser.parse_args()


def image_to_jpg_base64_url(image: np.ndarray | Image.Image):
    """Convert a numpy array to a base64 encoded image url."""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


def obs_preprocessor(obs: dict) -> dict:

    return {
        "chat_messages": obs["chat_messages"],
        # need to convert to base64 to work with llm-reasoners visualizer client
        # don't want a massive list
        "screenshot": image_to_jpg_base64_url(obs["screenshot"]),
        # "screenshot": image_to_jpg_base64_url(obs["screenshot"]),
        "goal_object": obs["goal_object"],
        "last_action": obs["last_action"],
        "last_action_error": obs["last_action_error"],
        "open_pages_urls": obs["open_pages_urls"],
        "open_pages_titles": obs["open_pages_titles"],
        "active_page_index": obs["active_page_index"],
        "axtree_txt": flatten_axtree_to_str(obs["axtree_object"]),
        "pruned_html": prune_html(flatten_dom_to_str(obs["dom_object"])),
    }


def get_scroll_position(page):
    return page.evaluate("""() => {
        const scrollTop = window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const remainingPixels = documentHeight - (scrollTop + windowHeight);

        return {
            'scrollTop': scrollTop,
            'windowHeight': windowHeight,
            'documentHeight': documentHeight,
            'remainingPixels': remainingPixels
        };
    }""")


def get_serializable_obs(env, obs):
    scroll_position = get_scroll_position(env.page)
    obs['scroll_position'] = scroll_position
    # make observation serializable
    obs['screenshot'] = image_to_jpg_base64_url(obs['screenshot'])
    obs['active_page_index'] = obs['active_page_index'].item()
    obs['elapsed_time'] = obs['elapsed_time'].item()
    return obs


class ParseError(Exception):
    pass


def parser(text, keys, optional_keys=()):
    try:
        ans_dict = parse_html_tags_raise(text, keys, optional_keys)
    except ParseError as e:
        return None, False, str(e)
    return ans_dict, True, ''


def extract_html_tags(text, keys):
    """Extract the content within HTML tags for a list of keys.

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.

    Returns
    -------
    dict
        A dictionary mapping each key to a list of subset in `text` that match the key.

    Notes
    -----
    All text and keys will be converted to lowercase before matching.

    """
    content_dict = {}
    # text = text.lower()
    # keys = set([k.lower() for k in keys])
    for key in keys:
        pattern = f'<{key}>(.*?)</{key}>'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            content_dict[key] = [match.strip() for match in matches]
    return content_dict


def parse_html_tags_raise(text, keys=(), optional_keys=(), merge_multiple=False):
    """A version of parse_html_tags that raises an exception if the parsing is not successful."""
    content_dict, valid, retry_message = parse_html_tags(
        text, keys, optional_keys, merge_multiple=merge_multiple
    )
    if not valid:
        raise ParseError(retry_message)
    return content_dict


def parse_html_tags(text, keys=(), optional_keys=(), merge_multiple=False):
    """Satisfy the parse api, extracts 1 match per key and validates that all keys are present

    Parameters
    ----------
    text : str
        The input string containing the HTML tags.
    keys : list of str
        The HTML tags to extract the content from.
    optional_keys : list of str
        The HTML tags to extract the content from, but are optional.

    Returns
    -------
    dict
        A dictionary mapping each key to subset of `text` that match the key.
    bool
        Whether the parsing was successful.
    str
        A message to be displayed to the agent if the parsing was not successful.
    """
    all_keys = tuple(keys) + tuple(optional_keys)
    content_dict = extract_html_tags(text, all_keys)
    retry_messages = []

    for key in all_keys:
        if key not in content_dict:
            if key not in optional_keys:
                retry_messages.append(
                    f'Missing the key <{key}> in the answer.')
        else:
            val = content_dict[key]
            content_dict[key] = val[0]
            if len(val) > 1:
                if not merge_multiple:
                    retry_messages.append(
                        f'Found multiple instances of the key {key}. You should have only one of them.'
                    )
                else:
                    # merge the multiple instances
                    content_dict[key] = '\n'.join(val)

    valid = len(retry_messages) == 0
    retry_message = '\n'.join(retry_messages)
    return content_dict, valid, retry_message


def get_scroll_position(page):
    return page.evaluate("""() => {
        const scrollTop = window.scrollY;
        const windowHeight = window.innerHeight;
        const documentHeight = document.documentElement.scrollHeight;
        const remainingPixels = documentHeight - (scrollTop + windowHeight);

        return {
            'scrollTop': scrollTop,
            'windowHeight': windowHeight,
            'documentHeight': documentHeight,
            'remainingPixels': remainingPixels
        };
    }""")


def get_serializable_obs(env, obs):
    scroll_position = get_scroll_position(env.page)
    obs['scroll_position'] = scroll_position
    # make observation serializable
    # obs["screenshot"] = overlay_som(screenshot=obs["screenshot"], extra_properties=)

    obs["screenshot_som"] = overlay_som(
        obs["screenshot"], extra_properties=obs["extra_element_properties"]
    )
    obs["screenshot_som_base64"] = image_to_jpg_base64_url(
        obs["screenshot_som"])

    # print(obs["screenshot_som_base64"])

    # save som image
    # print(type(obs["screenshot_som"]))
    # save numpy array to png
    # Image.fromarray(obs["screenshot_som"]).save("screenshot_som___.png")

    obs['screenshot'] = image_to_jpg_base64_url(obs['screenshot'])
    obs['active_page_index'] = obs['active_page_index'].item()
    obs['elapsed_time'] = obs['elapsed_time'].item()
    return obs
