import enum
import os
import platform


class ExecutionEnvironment(enum.Enum):
    DEV = "dev"
    PROD = "prod"
    LOCAL = "local"
    STAGING = "staging"


class ExecutionEnvironmentManager:
    @staticmethod
    def get_execution_environment() -> ExecutionEnvironment:
        if platform.system() == "Windows":
            return ExecutionEnvironment.LOCAL
        if os.getenv("AZURE_ENVIRONMENT_NAME") is None:
            raise EnvironmentError("AZURE_ENVIRONMENT_NAME not set")
        return (
            ExecutionEnvironment.PROD
            if os.getenv("AZURE_ENVIRONMENT_NAME") == "prod"
            else ExecutionEnvironment.DEV
        )
