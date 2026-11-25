import models as core

def check(value, code):
    if not value:
        raise core.ProtocolError(code)
