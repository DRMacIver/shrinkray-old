from hypothesis import settings, HealthCheck

settings.register_profile(
    "default", settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50,)
)

settings.load_profile("default")
