document.addEventListener("DOMContentLoaded", () => {
    const chatMessages = document.getElementById("chat-messages");
    const userInput = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const themeToggleBtn = document.getElementById("theme-toggle-btn");

    // Theme toggle functionality
    let isLightMode = false;
    themeToggleBtn.addEventListener("click", () => {
        isLightMode = !isLightMode;
        document.body.classList.toggle("light-mode", isLightMode);

        // Update button text
        if (isLightMode) {
            themeToggleBtn.textContent = "🌙 Dark Mode";
        } else {
            themeToggleBtn.textContent = "☀️ Light Mode";
        }
    });

    // Flag to track whether a request is in progress
    let isProcessing = false;

    async function sendMessage() {
        if (isProcessing) return; // Prevent multiple requests
        isProcessing = true;

        const message = userInput.value.trim();
        if (!message) {
            isProcessing = false; // Reset the flag if the input is empty
            return;
        }

        try {
            // Disable the send button immediately
            sendBtn.disabled = true;

            // Add user message to the chat
            appendMessage(message, "user");
            userInput.value = "";

            // Simulate a delay to test the button's disabled state
            // await new Promise((resolve) => setTimeout(resolve, 2000));

            // Send the message to the Flask backend
            const response = await fetch("/generate", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ message }),
            });

            // Parse the response
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // Add bot response to the chat
            appendMessage(data.response, "bot");
        } catch (error) {
            appendMessage(
                "Sorry, there was an error processing your request.",
                "bot"
            );
        } finally {
            // Re-enable the send button after processing
            sendBtn.disabled = false;
            isProcessing = false; // Allow new requests
        }
    }

    function appendMessage(content, sender) {
        const messageDiv = document.createElement("div");
        if (sender === "user") {
            // User messages remain in chat bubbles
            messageDiv.className = `message user-message`;
            messageDiv.textContent = content;
        } else if (sender === "bot") {
            // Bot responses are rendered as raw HTML
            messageDiv.className = "bot-response";
            messageDiv.innerHTML = content; // Use innerHTML to render HTML content
        }
        chatMessages.appendChild(messageDiv);

        // Auto-scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Event listeners
    sendBtn.addEventListener("click", sendMessage);
    userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
});