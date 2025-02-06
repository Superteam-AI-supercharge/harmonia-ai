import logging
import os
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, ConversationHandler,
    ContextTypes, filters
)
import requests

# URL of the FastAPI endpoint for queries (your existing endpoint)
API_URL = "http://localhost:8000/query"

# Admin user IDs (replace with actual admin Telegram user IDs)
ADMIN_IDS = [123456789, 987654321]  # Example IDs

# Define conversation states
SELECT_DIR, WAIT_DOCUMENT = range(2)

# List of available directories for upload
UPLOAD_DIRECTORIES = [
    "superteam",
    "solana",
    "superteam_vn",
    "superteam_nig",
    "bounties",
    "grants",
    "hackatons"
]

# Set up logging.
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a welcome message when the bot starts."""
    await update.message.reply_text("Hello! I am the Superteam Knowledge Bot. Ask me anything.")

async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming text messages as queries."""
    user_query = update.message.text
    try:
        # Post the query to the FastAPI endpoint.
        response = requests.post(API_URL, json={"query": user_query})
        if response.status_code == 200:
            data = response.json()
            answer = data.get("answer", "NO")
            await update.message.reply_text(answer)
        else:
            await update.message.reply_text("Error retrieving answer.")
    except Exception as e:
        await update.message.reply_text("An error occurred while processing your query.")

# --- Admin Upload Handlers ---
async def admin_upload_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start the upload conversation; only admins can use this."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("You are not authorized to use this feature.")
        return ConversationHandler.END

    # Provide the list of directories for upload.
    keyboard = [[directory] for directory in UPLOAD_DIRECTORIES]
    reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
    await update.message.reply_text(
        "Please choose the target directory for the document upload:",
        reply_markup=reply_markup
    )
    return SELECT_DIR

async def select_directory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Save the selected directory and ask for the document upload."""
    directory = update.message.text.strip()
    if directory not in UPLOAD_DIRECTORIES:
        await update.message.reply_text("Invalid directory. Please try again.")
        return SELECT_DIR

    # Save selected directory in context
    context.user_data['upload_directory'] = directory
    await update.message.reply_text(
        f"You have selected '{directory}'. Now, please upload the document.",
        reply_markup=ReplyKeyboardRemove()
    )
    return WAIT_DOCUMENT

async def handle_document_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Process the uploaded document file."""
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text("You are not authorized to use this feature.")
        return ConversationHandler.END

    # Retrieve the selected directory from the conversation data.
    directory = context.user_data.get('upload_directory', 'default')
    document = update.message.document

    if not document:
        await update.message.reply_text("No document found. Please try again.")
        return WAIT_DOCUMENT

    # Download the file to a temporary location.
    file = await document.get_file()
    file_path = f"temp/{document.file_name}"
    os.makedirs("temp", exist_ok=True)
    await file.download_to_drive(file_path)

    # Here you can implement logic to move the file to the proper directory.
    # For demonstration, we simply reply with the chosen directory and file name.
    # You might call your backend upload endpoint here.
    await update.message.reply_text(f"File '{document.file_name}' uploaded to '{directory}' directory.")

    # Clean up temporary file if desired.
    os.remove(file_path)

    return ConversationHandler.END

async def cancel_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Cancel the upload conversation."""
    await update.message.reply_text("Upload cancelled.", reply_markup=ReplyKeyboardRemove())
    return ConversationHandler.END

def main():
    """Start the Telegram bot."""
    # Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your actual Telegram bot token.
    application = ApplicationBuilder().token("8084141502:AAF1vX9QnBBJQdvmxRIA6s-V_LPgPMPYnrg").build()

    # Add the start and query handlers.
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

    # Set up the conversation handler for admin uploads.
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






# # telegram_bot.py
# import logging
# import requests
# from telegram import Update
# from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes

# # URL of the FastAPI endpoint we created in Step 2.
# API_URL = "http://localhost:8000/query"

# # Set up logging.
# logging.basicConfig(
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
# )

# async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Send a welcome message when the bot starts."""
#     await update.message.reply_text("Hello! I am the Superteam Knowledge Bot. Ask me anything.")

# async def handle_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
#     """Handle incoming text messages as queries."""
#     user_query = update.message.text
#     try:
#         # Post the query to the FastAPI endpoint.
#         response = requests.post(API_URL, json={"query": user_query})
#         if response.status_code == 200:
#             data = response.json()
#             answer = data.get("answer", "NO")
#             await update.message.reply_text(answer)
#         else:
#             await update.message.reply_text("Error retrieving answer.")
#     except Exception as e:
#         await update.message.reply_text("An error occurred while processing your query.")

# def main():
#     """Start the Telegram bot."""
#     # Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your actual Telegram bot token.
#     application = ApplicationBuilder().token("8084141502:AAF1vX9QnBBJQdvmxRIA6s-V_LPgPMPYnrg").build()

#     # Handlers for commands and text messages.
#     application.add_handler(CommandHandler("start", start))
#     application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query))

#     # Start polling for updates from Telegram.
#     application.run_polling()

# if __name__ == '__main__':
#     main()



