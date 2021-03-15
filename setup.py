import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ByeongChatbot", # Replace with your own username
    version="0.0.1",
    author="Byeongryul",
    author_email="mapia56@naver.com",
    description="this is a chatbot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Byeongryul/Book_Chatbot.git",
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    package_dir={"": "/usr/lib/python3/dist-packages"},
    packages=setuptools.find_packages(where="/usr/lib/python3/dist-packages"),
    python_requires=">=3.6",
)