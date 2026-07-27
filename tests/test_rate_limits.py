from rap_mixer.security.rate_limits import UsageLimiter


def test_owner_limit_enforced():
    limiter = UsageLimiter(max_session=2, max_per_minute=2)
    assert limiter.consume() and limiter.consume()
    assert not limiter.consume()

