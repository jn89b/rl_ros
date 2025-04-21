"""
Example class for the package.
"""


class Drone:
    def __init__(self,
                 name: str):
        self.name: str = name

    def go_brr(self) -> None:
        print(self.name + " goes brr")
