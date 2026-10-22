class ServiceError(Exception):
    """Base class; `status` is the matching HTTP status code."""
    status = 400


class InvalidRequest(ServiceError):
    status = 400


class NotAllowed(ServiceError):
    status = 403


class NotFound(ServiceError):
    status = 404


class Conflict(ServiceError):
    status = 409
