from langchain.tools import tool


@tool
def get_brightness() -> str:
    """
    Get the brightness of the light in the room.
    """
    return "The brightness of the light is 5."


@tool
def adjust_light(brightness: int) -> str:
    """
    Adjust the brightness of the light in the room.

    Args:
        brightness: The brightness of the light. 1-10 is the brightness.

    Returns:
        A message indicating that the light has been adjusted.
    """
    print(f"Adjusting light to {brightness}%")
    return f"Light adjusted to {brightness}%"
