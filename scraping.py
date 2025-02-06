import re
import json

def extract_header(lines):
    """
    Extract title, author, and publish date from the first three non-empty lines.
    If any are missing, use a fallback value.
    """
    header = []
    for line in lines:
        stripped = line.strip()
        if stripped:
            header.append(stripped)
        if len(header) == 3:
            break
    # Provide fallback values if not enough header lines are found.
    title = header[0] if len(header) > 0 else "Untitled"
    author = header[1] if len(header) > 1 else "Unknown Author"
    publish_date = header[2] if len(header) > 2 else "Unknown Date"
    return title, author, publish_date


def extract_urls(text):
    """
    Extract all URLs from the text.
    """
    url_pattern = r'(https?://[^\s]+)'
    return re.findall(url_pattern, text)

def extract_images(lines):
    """
    Extract image information. This function looks for the word 'Image' in a line,
    and then checks the following line for a URL.
    """
    images = []
    for i, line in enumerate(lines):
        if "Image" in line:
            # If the next non-empty line is a URL, we capture it.
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                url_candidate = lines[j].strip()
                if re.match(r'https?://', url_candidate):
                    images.append({
                        "url": url_candidate,
                        "caption": "Image"  # Optionally, you can extract a caption if available.
                    })
    return images

def convert_txt_to_json(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Extract header information
    title, author, publish_date = extract_header(lines)
    
    # Assume content starts after the header (we skip the header lines)
    content_lines = lines[3:]
    content = "".join(content_lines).strip()
    
    # Extract URLs from the full content
    links = extract_urls(content)
    
    # Extract image URLs using a heuristic (lines with 'Image')
    images = extract_images(content_lines)

    # Build the JSON structure
    post = {
        "title": title,
        "author": author,
        "publish_date": publish_date,
        "category": "Newsletter",  # You can adjust this as needed.
        "content": content,
        "links": links,
        "images": images
    }

    # Save JSON to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(post, f, indent=4, ensure_ascii=False)

    print(f"Converted '{input_file}' to '{output_file}' successfully.")

if __name__ == "__main__":
    input_filename = "sample.txt"   # Your scraped text file
    output_filename = "output.json"   # The desired output JSON file name
    convert_txt_to_json(input_filename, output_filename)
