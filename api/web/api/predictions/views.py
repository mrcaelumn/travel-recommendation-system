from typing import List

from fastapi import APIRouter
from api.web.api.echo.schema import Message
from fastapi.param_functions import Depends

router = APIRouter()

@router.post("/", response_model=Message)
async def send_echo_message(
    incoming_message: Message,
) -> Message:
    """
    Sends echo back to user.

    :param incoming_message: incoming message.
    :returns: message same as the incoming.
    """
    return incoming_message


