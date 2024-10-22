// components/ChatWindow.tsx
"use client";

import React, { useState, useEffect } from "react";
type ChatWindowProps = {
  id: string; // Accept the id as a prop
};

const ChatWindow = ({ id }: ChatWindowProps) => {
  const [messages, setMessages] = useState<{ user: string; text: string }[]>(
    []
  );
  const [inputValue, setInputValue] = useState("");
  const [socket, setSocket] = useState<WebSocket | null>(null); // Store the WebSocket connection

  // Initialize WebSocket connection when the component mounts
  useEffect(() => {
    const newSocket = new WebSocket("ws://127.0.0.1:8000/ws"); // Update with your WebSocket URL
    setSocket(newSocket);

    newSocket.onopen = () => {
      newSocket.send(JSON.stringify({ id: id }));
    };

    newSocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log(data);
      setMessages((prevMessages) => [
        ...prevMessages,
        { user: "Bot", text: data.llm_response }, // Adjust according to your server response structure
      ]);
    };

    return () => {
      newSocket.close(); // Close the WebSocket connection on unmount
    };
  }, []);

  const handleSendMessage = () => {
    if (inputValue.trim() === "") return;

    // Send the message to the backend
    if (socket) {
      socket.send(JSON.stringify({ message: inputValue })); // Send the user message
    }

    // Update the local state with the user's message
    setMessages((prevMessages) => [
      ...prevMessages,
      { user: "User", text: inputValue },
    ]);
    setInputValue(""); // Clear the input field
    // };

    // // Mock response from the server (replace this with actual response handling)
    // setTimeout(() => {
    //   setMessages((prevMessages) => [
    //     ...prevMessages,
    //     { user: "Bot", text: "This is a response to: " + inputValue },
    //   ]);
    // }, 1000);
  };

  return (
    <div className="chat-window">
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.user.toLowerCase()}`}>
            <strong>{msg.user}:</strong> {msg.text}
          </div>
        ))}
      </div>
      <div className="chat-input">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="Type a message..."
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              handleSendMessage(); // Trigger send on Enter
            }
          }}
        />
        <button onClick={handleSendMessage}>Send</button>
      </div>

      <style jsx>{`
        .chat-window {
          position: fixed;
          bottom: 20px;
          right: 20px;
          width: 450px;
          max-height: 650px;
          border-radius: 12px;
          background-color: #f8f9fa;
          display: flex;
          flex-direction: column;
          overflow: hidden;
          box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
          transition: all 0.3s ease-in-out;
        }

        .chat-window:hover {
          box-shadow: 0 6px 30px rgba(0, 0, 0, 0.2);
        }

        .chat-messages {
          flex: 1;
          padding: 15px;
          overflow-y: auto;
          max-height: 350px;
          border-radius: 0 0 12px 12px;
          background-color: #ffffff;
        }

        .message {
          margin: 8px 0;
          padding: 8px 12px;
          border-radius: 20px;
          max-width: 80%;
        }

        .user {
          background-color: #282a3a;
          color: white;
          align-self: flex-end;
        }

        .bot {
          background-color: #f3f4f8;
          color: #333;
          align-self: flex-start;
        }

        .chat-input {
          display: flex;
          border-top: 1px solid #ddd;
          padding: 10px;
          background-color: #f1f1f1;
        }

        .chat-input input {
          flex: 1;
          padding: 10px 12px;
          border: none;
          border-radius: 20px;
          margin-right: 10px;
          outline: none;
          box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
        }

        .chat-input button {
          padding: 10px 16px;
          border: none;
          border-radius: 35px;
          background-color: #282a3a;
          color: white;
          cursor: pointer;
          transition: background-color 0.3s;
          font-weight: bold;
        }

        .chat-input button:hover {
          background-color: #005bb5;
        }
      `}</style>
    </div>
  );
};

export default ChatWindow;
