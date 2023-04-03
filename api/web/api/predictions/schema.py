from models.rbm import RBM


class RBMModelDTO(RBM):
    """
    DTO for dummy models.

    It returned when accessing dummy models from the API.
    """

    id: int

    class Config:
        orm_mode = True
