from collections import deque
from dataclasses import dataclass, field
from time import monotonic


@dataclass
class UsageLimiter:
    max_session: int = 20
    max_per_minute: int = 5
    requests: deque = field(default_factory=deque)
    total: int = 0

    def consume(self) -> bool:
        now = monotonic()
        while self.requests and self.requests[0] < now - 60:
            self.requests.popleft()
        if self.total >= self.max_session or len(self.requests) >= self.max_per_minute:
            return False
        self.requests.append(now)
        self.total += 1
        return True

