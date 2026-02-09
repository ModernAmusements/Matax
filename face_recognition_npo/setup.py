from setuptools import setup, find_packages
import os

# Get the current directory
cwd = os.path.abspath(os.path.dirname(__file__))

# Read the README file for long description
with open(os.path.join(cwd, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read the requirements file
with open(os.path.join(cwd, "requirements.txt"), encoding="utf-8") as f:
    install_requires = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="face-recognition-npo",
    version="0.1.0",
    description="NGO Facial Image Analysis System for Documentation Verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="NGO Technical Team",
    author_email="tech@ngo.org",
    url="https://github.com/ngo/face-recognition-npo",
    packages=find_packages(exclude=["tests*", "examples*", "utils*"]),
    package_data={
        "": ["*.txt", "*.md"],
    },
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=19.0",
            "flake8>=3.0",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "face-recognition-npo=main:main",
        ],
    },
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Other Audience",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Security",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Other/Nonlisted Topic",
    ],
    keywords=[
        "face recognition",
        "facial analysis",
        "ngo",
        "documentation",
        "verification",
        "ethics",
        "privacy",
        "human rights",
    ],
    project_urls={
        "Bug Reports": "https://github.com/ngo/face-recognition-npo/issues",
        "Source": "https://github.com/ngo/face-recognition-npo",
    },
    zip_safe=False,
)