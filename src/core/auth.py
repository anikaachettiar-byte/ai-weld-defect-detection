"""Authentication and Role-Based Access Control (RBAC) for NDT weld inspection."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class UserProfile:
    """Certified NDT inspector profile adhering to ISO 9712 and ASNT SNT-TC-1A standards."""
    username: str
    full_name: str
    role: str
    cert_id: str
    facility: str
    permissions: List[str]


# Pre-configured demo accounts for academic and industrial evaluation
DEFAULT_USERS: Dict[str, dict] = {
    "inspector": {
        "password": "ndt123",
        "profile": UserProfile(
            username="inspector",
            full_name="Vikram Sharma",
            role="Level II NDT Radiographer",
            cert_id="ASNT-RT-5491",
            facility="Boiler & Pressure Vessel Shop #3",
            permissions=["upload", "preprocess", "inspect", "sign_off"],
        ),
    },
    "qa_manager": {
        "password": "admin123",
        "profile": UserProfile(
            username="qa_manager",
            full_name="Dr. Elena Rostova",
            role="Level III QA Welding Supervisor",
            cert_id="ISO-9712-RT-Level-III",
            facility="Quality Assurance Division",
            permissions=["upload", "preprocess", "inspect", "sign_off", "override", "audit"],
        ),
    },
    "guest": {
        "password": "demo",
        "profile": UserProfile(
            username="guest",
            full_name="Demo Radiographer",
            role="Guest Trainee Inspector",
            cert_id="TRAINEE-RT-001",
            facility="NDT Evaluation Lab",
            permissions=["upload", "preprocess", "inspect"],
        ),
    },
}


def authenticate(username: str, password: str) -> Optional[UserProfile]:
    """Verify credentials and return user profile if valid."""
    u = username.strip().lower()
    if u in DEFAULT_USERS and DEFAULT_USERS[u]["password"] == password.strip():
        return DEFAULT_USERS[u]["profile"]
    return None


def get_demo_user(username: str) -> Optional[UserProfile]:
    """Retrieve demo user profile directly for one-click demo login."""
    u = username.strip().lower()
    if u in DEFAULT_USERS:
        return DEFAULT_USERS[u]["profile"]
    return None
