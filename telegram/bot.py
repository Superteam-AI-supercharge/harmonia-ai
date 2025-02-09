import logging
import os
import requests
import shutil
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ConversationHandler,
    ContextTypes,
    filters
)
from dotenv import load_dotenv

load_dotenv()

# Define conversation states for upload
SELECT_DIR, WAIT_DOCUMENT = range(2)

# Define API URL for your backend endpoints
api_url = "http://localhost:8000"  # Update as needed

# Telegram Bot Token
telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")

# Define Admin IDs and available upload directories (update these accordingly)
ADMIN_IDS = [123456789]  # Replace with actual admin Telegram user IDs
UPLOAD_DIRECTORIES = ["superteam", "solana", "superteam_vn", "superteam_nig", "bounties", "grants", "hackatons"]

# Set up logging.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ---------------------------
# Command Handlers
# ---------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the bot starts."""
    await update.message.reply_text("Hello! I am the Superteam Knowledge Bot. Use /help to see available commands.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Fallback: Handle incoming text messages as general queries."""
    user_query = update.message.text
    try:
        response = requests.post(api_url, json={"query": user_query})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "NO")
            await update.message.reply_text(answer)
        else:
            await update.message.reply_text(f"Error retrieving answer: {response.status_code}")
    except Exception as e:
        await update.message.reply_text("An error occurred while processing your query: " + str(e))

# ---------------------------
# /help Command and Callback Handler
# ---------------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a list of available commands as an inline keyboard."""
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/find <query> - Find Superteam members\n"
        "/learn <query> - Ask general questions about Superteam\n"
        "/upload - Upload a document (admin only)\n"
        "/delete <file_name> - Delete document by file name (admin only)\n"
        "/list - List all uploaded documents (admin only)\n\n"
        "Click a command below for usage instructions."
    )
    keyboard = [
        [InlineKeyboardButton("/find", callback_data="help_find")],
        [InlineKeyboardButton("/learn", callback_data="help_learn")],
        [InlineKeyboardButton("/upload", callback_data="help_upload")],
        [InlineKeyboardButton("/delete", callback_data="help_delete")],
        [InlineKeyboardButton("/list", callback_data="help_list")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(help_text, reply_markup=reply_markup)

async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks from the /help command inline keyboard."""
    query = update.callback_query
    await query.answer()  # Acknowledge the callback
    instructions = {
        "help_find": "Usage: /find <query>\nExample: /find skills in superteam_vietnam_members",
        "help_learn": "Usage: /learn <query>\nExample: /learn What is Superteam?",
        "help_upload": "Usage: /upload\nThen follow prompts to select a directory and upload a file. (Admin only)",
        "help_delete": "Usage: /delete <file_name>\nExample: /delete Youth Ambassador Program Proposal.pdf (Admin only)",
        "help_list": "Usage: /list\nLists all uploaded files, grouped by file name. (Admin only)"
    }
    data = query.data
    response_text = instructions.get(data, "No instructions available.")
    await query.edit_message_text(text=response_text)

# ---------------------------
# /find Command Handler
# ---------------------------
class FindRequest(BaseModel):
    query: str

async def find_members_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /find command by sending the query to the backend find endpoint."""
    text = update.message.text
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /find <query>")
        return

    query = parts[1]
    try:
        response = requests.post(f"http://localhost:8000/find", json={"query": query})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            context_str = data.get("context", "")
            await update.message.reply_text(f"Answer:\n{answer}\n\nContext:\n{context_str}")
        else:
            await update.message.reply_text(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# ---------------------------
# /learn Command Handler
# ---------------------------
class LearnRequest(BaseModel):
    query: str
    session_id: str = "default"

async def learn_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /learn command by sending the query to the backend learn endpoint."""
    text = update.message.text
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /learn <query>")
        return

    query = parts[1]
    payload = {"query": query, "session_id": "default"}
    try:
        response = requests.post(f"http://localhost:8000/learn", json=payload)
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            context_str = data.get("context", "")
            await update.message.reply_text(f"Answer:\n{answer}\n\nContext:\n{context_str}")
        else:
            await update.message.reply_text(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

# ---------------------------
# Admin Upload Handlers (Conversation for Upload)
# ---------------------------
async def admin_upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the upload conversation; only admins can use this."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("You are not authorized to use this feature.")
        return ConversationHandler.END

    keyboard = [[directory] for directory in UPLOAD_DIRECTORIES]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text("Please choose the target directory for the document upload:", reply_markup=reply_markup)
    return SELECT_DIR

async def select_directory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save the selected directory and ask for the document upload."""
    directory = update.message.text.strip()
    if directory not in UPLOAD_DIRECTORIES:
        await update.message.reply_text("Invalid directory. Please try again.")
        return SELECT_DIR

    context.user_data["upload_directory"] = directory
    await update.message.reply_text(f"You have selected '{directory}'. Now, please upload the document.", reply_markup=ReplyKeyboardRemove())
    return WAIT_DOCUMENT

async def handle_document_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process the uploaded document file."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("You are not authorized to use this feature.")
        return ConversationHandler.END

    directory = context.user_data.get("upload_directory", "default")
    document = update.message.document
    if not document:
        await update.message.reply_text("No document found. Please try again.")
        return WAIT_DOCUMENT

    file = await document.get_file()
    file_path = f"temp/{document.file_name}"
    os.makedirs("temp", exist_ok=True)
    await file.download_to_drive(file_path)

    # Move the file to permanent storage
    os.makedirs("uploads", exist_ok=True)
    permanent_path = os.path.join("uploads", document.file_name)
    shutil.copy(file_path, permanent_path)

    await update.message.reply_text(f"File '{document.file_name}' uploaded to '{directory}' directory and saved permanently.")

    os.remove(file_path)
    return ConversationHandler.END

async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the upload conversation."""
    await update.message.reply_text("Upload cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

# ---------------------------
# Admin Delete Command (/delete)
# ---------------------------
async def delete_file_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /delete command to remove documents by file name (admin only)."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("You are not authorized to use this feature.")
        return

    text = update.message.text
    parts = text.split(maxsplit=1)
    if len(parts) < 2:
        await update.message.reply_text("Usage: /delete <file_name>")
        return
    file_name = parts[1].strip()
    try:
        response = requests.post(f"{api_url.replace('/learn', '')}/delete_by_file", json={"admin_token": ADMIN_TOKEN, "file_name": file_name})
        if response.status_code == 200:
            await update.message.reply_text(f"File '{file_name}' deleted successfully.")
        else:
            await update.message.reply_text(f"Error deleting file: {response.status_code} - {response.text}")
    except Exception as e:
        await update.message.reply_text("An error occurred while deleting the file: " + str(e))

# ---------------------------
# Admin List Documents Command (/list)
# ---------------------------
async def list_documents_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle the /list command to show uploaded documents (grouped by file name)."""
    try:
        response = requests.get(f"{api_url.replace('/learn', '')}/list", params={"admin_token": ADMIN_TOKEN})
        if response.status_code == 200:
            data = response.json()
            docs = data.get("documents", [])
            if docs:
                reply_text = "Uploaded Files:\n"
                for doc in docs:
                    reply_text += f"- {doc['file_name']} (Directory: {doc['directory']})\n"
                await update.message.reply_text(reply_text)
            else:
                await update.message.reply_text("No documents found.")
        else:
            await update.message.reply_text(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        await update.message.reply_text("An error occurred while listing documents: " + str(e))

# ---------------------------
# /help Command and Callback Handler
# ---------------------------
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a list of available commands with an inline keyboard."""
    help_text = (
        "Available commands:\n"
        "/start - Start the bot\n"
        "/find <query> - Find Superteam members\n"
        "/learn <query> - Ask a general question about Superteam\n"
        "/upload - Upload a document (admin only)\n"
        "/delete <file_name> - Delete document by file name (admin only)\n"
        "/list - List all uploaded documents (admin only)\n\n"
        "Click a command below for usage instructions."
    )
    keyboard = [
        [InlineKeyboardButton("/find", callback_data="help_find")],
        [InlineKeyboardButton("/learn", callback_data="help_learn")],
        [InlineKeyboardButton("/upload", callback_data="help_upload")],
        [InlineKeyboardButton("/delete", callback_data="help_delete")],
        [InlineKeyboardButton("/list", callback_data="help_list")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(help_text, reply_markup=reply_markup)

async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries from the help inline keyboard."""
    query = update.callback_query
    await query.answer()  # Acknowledge callback
    instructions = {
        "help_find": "Usage: /find <query>\nExample: /find skills in superteam_vietnam_members",
        "help_learn": "Usage: /learn <query>\nExample: /learn What is Superteam?",
        "help_upload": "Usage: /upload\nThen follow prompts to select a directory and upload a file. (Admin only)",
        "help_delete": "Usage: /delete <file_name>\nExample: /delete Youth Ambassador Program Proposal.pdf (Admin only)",
        "help_list": "Usage: /list\nLists all uploaded files, grouped by file name. (Admin only)"
    }
    data = query.data
    response_text = instructions.get(data, "No instructions available.")
    await query.edit_message_text(text=response_text)

# ---------------------------
# Main function to start the bot
# ---------------------------
def main():
    application = ApplicationBuilder().token(telegram_bot_token).build()

    # Start command
    application.add_handler(CommandHandler("start", start))
    
    # /help command
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(help_callback, pattern="^help_"))
    
    # General query handler (fallback for any text that doesn't match a command)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))
    
    # Command handler for /find
    application.add_handler(CommandHandler("find", find_members_command))
    
    # Command handler for /learn
    application.add_handler(CommandHandler("learn", learn_command))
    
    # Command handler for /delete
    application.add_handler(CommandHandler("delete", delete_file_command))
    
    # Command handler for /list
    application.add_handler(CommandHandler("list", list_documents_command))
    
    # Conversation handler for admin uploads (/upload command)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("upload", admin_upload_start)],
        states={
            SELECT_DIR: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_directory)],
            WAIT_DOCUMENT: [MessageHandler(filters.Document.ALL, handle_document_upload)],
        },
        fallbacks=[CommandHandler("cancel", cancel_upload)]
    )
    application.add_handler(conv_handler)

    # Start polling for updates from Telegram.
    application.run_polling()

if __name__ == '__main__':
    main()
