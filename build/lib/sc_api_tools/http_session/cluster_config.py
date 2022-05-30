from dataclasses import dataclass

API_PATTERN = "/api/v1.0/"


@dataclass
class ClusterConfig:
    """
    Configuration for requests sessions, with host, username and password.
    """

    host: str
    username: str
    password: str

    @property
    def base_url(self) -> str:
        """
        Returns the base UR for accessing the cluster
        """
        return f"{self.host}{API_PATTERN}"
