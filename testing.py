import json

def remove_fields(data):
  """Removes the 'image_url' and 'username' fields from a list of dictionaries.

  Args:
    data: A list of dictionaries.

  Returns:
    A list of dictionaries with the specified fields removed.
  """
  for item in data:
    if 'image_url' in item:
      del item['image_url']
    if 'username' in item:
      del item['username']
  return data

# Load data from JSON file
with open('output.json', 'r') as f:
  data = json.load(f)

# Remove fields
modified_data = remove_fields(data)

# Write modified data back to JSON file (optional)
with open('modified.json', 'w') as f:
  json.dump(modified_data, f, indent=2)