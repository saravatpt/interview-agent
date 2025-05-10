# Web Application

This directory contains the files for a simple web application built using Python for the backend and HTML/JavaScript/CSS for the frontend.

## Project Structure

- `backend_server.py`: The backend server script (Python)
  - Handles API endpoints
  - Manages interview agent interactions
  - Processes requests and responses

- `index.html`: The main HTML file for the web page
  - Contains the application structure
  - Interview interface elements
  - User interaction components

- `script.js`: The JavaScript file for frontend functionality
  - Manages user interactions
  - Handles API calls to backend
  - Updates UI dynamically

- `style.css`: The CSS file for styling the web page
  - Defines layout and appearance
  - Responsive design elements
  - Theme and visual components

## Setup and Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start the backend server:
```bash
python backend_server.py
```

3. Open `index.html` in a web browser

## Usage

1. Open the application in your web browser
2. Start a new interview session
3. Interact with the AI interviewer
4. Review feedback and suggestions

## Development

- Backend runs on `http://localhost:5000` by default
- Frontend makes API calls to the backend endpoints
- Modify `style.css` for customizing appearance
- Update `script.js` for new frontend features

## Requirements

- Python 3.8+
- Modern web browser
- Internet connection for AI API access

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details
