from contextvars import ContextVar

# Holds the current request's ID. "-" means "not inside a request".
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
