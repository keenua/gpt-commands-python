from asyncio import run
from dataclasses import dataclass
from logging import getLogger
from typing import List, Optional

from dataclasses_jsonschema import JsonSchemaMixin

from gpt_commands import GPTCommandsClient


@dataclass
class Point(JsonSchemaMixin):
    "A 2D point"
    x: float
    y: float

@dataclass
class Marker(JsonSchemaMixin):
    "A marker"
    name: str
    point: Point

class Game:
    def __init__(self):
        self.markers: List[Marker] = []

    def get_inventory(self, character: str, max_items: int) -> List[str]:
        """
        Get inventory of a character

        Args:
            character (str): The name of the character to get the inventory of. One of: 'Harry', 'Ron', 'Hermione'
            max_items (int): The maximum number of items to return
        Returns:
            List[str]: The inventory of the character
        """
        if character == "Harry":
            return ["Wand", "Broom", "Cloak"]
        elif character == "Ron":
            return ["Wand", "Rat"]
        elif character == "Hermione":
            return ["Wand", "Cat", "Book"]

        return []

    def alohomora(self):
        """
        Unlock the door
        """
        print("[COMMAND] Alohomora!")

    def expelliarmus(self, target: str):
        """
        Disarm the target

        Args:
            target (str): The target to disarm
        """
        print(f"[COMMAND] Expelliarmus {target}!")

    def get_location_coordinates(self, location: Optional[str] = None) -> Point:
        """
        Get the coordinates of a location

        Args:
            location (Optional[str], optional): The location to get the coordinates of. One of: 'Hogwarts', 'Diagon Alley', 'Forbidden Forest'. If None, defaults to current location.

        Returns:
            Point: The coordinates of the location
        """

        if location == "Hogwarts":
            return Point(x=0, y=0)
        elif location == "Diagon Alley":
            return Point(x=1, y=1)
        elif location == "Forbidden Forest":
            return Point(x=2, y=2)

        return Point(x=100, y=100)
    
    def get_makers(self) -> List[Marker]:
        """
        Get the markers on the map

        Returns:
            List[Marker]: The markers
        """
        return self.markers

    def set_a_mark_on_the_map(self, marker: Marker):
        """
        Set a mark on the map

        Args:
            marker (Marker): The mark to set
        """
        print(f"[COMMAND] Set a mark on the map at {marker.point} with {marker.name}!")

        self.markers.append(marker)


async def main():
    logger = getLogger(__name__)
    logger.info("Starting ...")

    system_prompt = """
        You are Harry Potter, a wizard living in the Wizarding World. 
        You're in dangerous place along with your friend Neville. 
        You're talking to Neville as Harry and never break the character.
    """

    manager = Game()
    model = "gpt-4-0613"  # "gpt-3.5-turbo-16k-0613"
    async with GPTCommandsClient(model, system_prompt) as client:
        while True:
            prompt = input("You: ")
            async for data in client.chat_stream(prompt, manager):
                print(data, end="")
            print()


if __name__ == "__main__":
    run(main())
