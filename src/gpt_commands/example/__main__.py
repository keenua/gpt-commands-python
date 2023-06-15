from asyncio import run
from logging import getLogger
from typing import List

import gpt_commands.openai.utils
from gpt_commands.openai import GPTCommandsClient


class Game:
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
