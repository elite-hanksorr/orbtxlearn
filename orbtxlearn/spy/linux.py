import xdo

from . import spy

class LinuxSpy(spy.Spy):
    def __init__(self, *args, **kwargs) -> None:
        super(LinuxSpy, self).__init__(*args, **kwargs)
        self.__xdo = xdo.Xdo()  # type: ignore
        print('Please click on the OrbtXL window')
        self.__window = self.__xdo.select_window_with_click()

    def keydown(self) -> None:
        # TODO experiment with delay. Looking through libxdo, it seems that a delay isn't necessary
        self.__xdo.send_keysequence_window_down(self.__window, 'space', delay=0)

    def keyup(self) -> None:
        # TODO experiment with delay. Looking through libxdo, it seems that a delay isn't necessary
        self.__xdo.send_keysequence_window_up(self.__window, 'space', delay=0)
