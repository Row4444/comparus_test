import pytesseract
from PIL import Image
import re
import cv2
import argparse
from pathlib import Path


DEFAULT_INPUT_IMG_NAME: str = "emails.png"
DEFAULT_INPUT_IMG_FOLDER: str = "inputs"

DEFAULT_OUTPUT_IMG_NAME: str = "result.txt"
DEFAULT_OUTPUT_IMG_FOLDER: str = "outputs"

input_file_path: Path = Path(DEFAULT_INPUT_IMG_FOLDER) / DEFAULT_INPUT_IMG_NAME
output_file_path: Path = Path(DEFAULT_OUTPUT_IMG_FOLDER) / DEFAULT_OUTPUT_IMG_NAME


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(binary)
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return deskewed


def extract_emails_from_image(image_path: str) -> list[str]:
    image = cv2.imread(image_path)
    deskewed_image = deskew_image(image)

    gray_image = cv2.cvtColor(deskewed_image, cv2.COLOR_BGR2GRAY)
    _, processed_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    text: str = pytesseract.image_to_string(Image.fromarray(processed_image))
    email_pattern: str = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    emails: list[str] = re.findall(email_pattern, text)
    return emails


def save_emails_to_file(emails: list[str], output_path: str) -> None:
    with open(output_path, "w") as file:
        for email in emails:
            file.write(email + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(input_file_path),
        help="Input image",
    )
    parser.add_argument(
        "--output",
        default=str(output_file_path),
        help="Output .txt file",
    )
    args = parser.parse_args()

    emails: list = extract_emails_from_image(args.input)

    if emails:
        save_emails_to_file(emails, args.output)
        print(f"Output file: {args.output}")
    else:
        print("No e-mails found")


if __name__ == "__main__":
    main()
