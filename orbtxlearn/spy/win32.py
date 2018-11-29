import ctypes

from . import spy

__all__ = ['WindowsSpy']

LONG = ctypes.c_long
DWORD = ctypes.c_ulong
ULONG_PTR = ctypes.POINTER(DWORD)
WORD = ctypes.c_ushort

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ('dx', LONG),
        ('dy', LONG),
        ('mouseData', DWORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]

class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ('wVk', WORD),
        ('wScan', WORD),
        ('dwFlags', DWORD),
        ('time', DWORD),
        ('dwExtraInfo', ULONG_PTR)
    ]

class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ('uMsg', DWORD),
        ('wParamL', WORD),
        ('wParamH', WORD)
    ]

class union_INPUT(ctypes.Union):
    _fields_ = [
        ('mi', MOUSEINPUT),
        ('ki', KEYBDINPUT),
        ('hi', HARDWAREINPUT)
    ]

class LPINPUT(ctypes.Structure):
    _fields_ = [
        ('type', DWORD),
        ('union', union_INPUT)
    ]


class WindowsSpy(spy.Spy):
    def __init__(self, *args, **kwargs) -> None:
        super(WindowsSpy, self).__init__(*args, **kwargs)

    def keydown(self) -> None:
        # TODO experiment with delay. Looking through libxdo, it seems that a delay isn't necessary
        super(WindowsSpy, self).keydown()

    def keyup(self) -> None:
        # TODO experiment with delay. Looking through libxdo, it seems that a delay isn't necessary
        super(WindowsSpy, self).keyup()
