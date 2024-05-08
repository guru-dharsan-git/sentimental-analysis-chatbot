from tkinter import *
import threading
import pyttsx3
import json
import speech_recognition as sr
from chat import get_response, bot_name
from nltk.sentiment import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
sia = SentimentIntensityAnalyzer()

BG_GRAY = "#F0F0F0"
BG_COLOR = "#000000"
TEXT_COLOR = "#FFFFFF"

FONT = "Helvetica 14"
FONT_BOLD = "Helvetica 13 bold"

class ChatApplication:
    def __init__(self):
        self.window = Tk()
        self._setup_main_window()
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)
        self.recognizer = sr.Recognizer()

    def run(self):
        self.window.mainloop()

    def _setup_main_window(self):
        self.window.title("Chat now!!")
        self.window.geometry("680x680")
        self.window.resizable(width=False, height=False)
        self.window.configure(bg=BG_COLOR)
        
        self.bot_name = Label(self.window, bg=BG_COLOR, fg="red", pady=10)

        # Head Label
        head_label = Label(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                           text="Ask me anything...", font=FONT_BOLD, pady=10)
        head_label.pack(fill=X)
       
        # Text Widget
        self.text_widget = Text(self.window, bg=BG_COLOR, fg=TEXT_COLOR,
                                font=FONT, padx=5, pady=5)
        self.text_widget.pack(fill=BOTH, expand=True)
        self.text_widget.configure(cursor="arrow", state=DISABLED)

        # Scrollbar
        scrollbar = Scrollbar(self.text_widget)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.text_widget.yview)

        # Message Entry
        self.msg_entry = Entry(self.window, bg=BG_GRAY, fg=BG_COLOR, font=FONT)
        self.msg_entry.pack(fill=X, pady=10, padx=10)
        self.msg_entry.focus()
        self.msg_entry.bind("<Return>", self._on_enter_pressed)

        # Send Button
        send_button = Button(self.window, text="Send", font=FONT_BOLD, width=12,
                             bg=TEXT_COLOR, command=lambda: self._on_enter_pressed(None))
        send_button.pack(fill=X, padx=10, pady=10)

        # Listen Button
        listen_button = Button(self.window, text="Listen", font=FONT_BOLD, width=12,
                               bg=TEXT_COLOR, command=self._start_listening)
        listen_button.pack(fill=X, padx=10, pady=10)

    def _on_enter_pressed(self, event):
        msg = self.msg_entry.get()
        self._insert_message(msg, "You")
        self._speak_response(msg)

    def _speak_response(self, msg):
        response = get_response(msg)
        translated_response = self._translate_to_tamil(response)
        t = threading.Thread(target=self._speak_in_thread, args=(f"{translated_response}",))
        t.start()

    def _translate_to_tamil(self, text):
        try:
            translated_text = GoogleTranslator(source='auto', target='ta').translate(text)
            return translated_text
        except Exception as e:
            print(f"Translation Error: {e}")
            return text

    def _speak_in_thread(self, text):
        self.engine.say(text)
        self.engine.runAndWait()

    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        self.msg_entry.delete(0, END)
        msg1 = f"{sender}: {msg}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg1)
        self.text_widget.configure(state=DISABLED)
        
        sentiment_scores = sia.polarity_scores(msg)

        msg2 = f"{bot_name}: {get_response(msg)}\n{sia.polarity_scores(msg)}\n\n"
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg2, "bot")
        self.text_widget.tag_configure("bot", foreground="red")
        self.text_widget.configure(state=DISABLED)
        
        self.text_widget.see(END)

        self.save_sentiment_to_file(msg, sentiment_scores)
    def save_sentiment_to_file(self, message, sentiment_scores):
        data = {
            "message": message,
            "sentiment_scores": sentiment_scores
        }

        with open("sentiment_data.json", "a") as json_file:
            json.dump(data, json_file)
            json_file.write("\n")

    def _start_listening(self):
        with sr.Microphone() as source:
            self.msg_entry.delete(0, END)
            self.msg_entry.insert(0, "Listening...")
            self.msg_entry.update_idletasks()
            audio = self.recognizer.listen(source)
        
        try:
            user_input = self.recognizer.recognize_google(audio)
            self.msg_entry.delete(0, END)
            self.msg_entry.insert(0, user_input)
            self._on_enter_pressed(None)
        except sr.UnknownValueError:
            self.msg_entry.delete(0, END)
            self.msg_entry.insert(0, "Could not understand audio.")
        except sr.RequestError as e:
            self.msg_entry.delete(0, END)
            self.msg_entry.insert(0, f"Error: {e}")

if __name__ == "__main__":
    app = ChatApplication()
    app.run()
