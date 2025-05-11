import pytest
from app.main import create_app
from app.config.config import TestingConfig

@pytest.fixture
def app():
    app = create_app()
    app.config.from_object(TestingConfig)
    return app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner() 