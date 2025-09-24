const sendBtn = document.getElementById("sendBtn");
const input = document.getElementById("input");
const messagesDiv = document.getElementById("messages");

// Add a message to chat window
function addMessage(text, side = "right") {
  const msg = document.createElement("div");
  msg.className = "msg " + side;
  msg.textContent = text;
  messagesDiv.appendChild(msg);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Send message to backend
async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  addMessage(text, "right");
  input.value = "";

  const resp = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: text })
  });
  const data = await resp.json();

  addMessage(data.reply, "left");

  // 🔗 Auto redirect if navigation is detected
  if (data.navigate) {
    setTimeout(() => {
      // window.location.href = data.navigate;
      window.open(data.navigate, "_blank")
    }, 1500); // 1.5 sec so user sees the bot response
  }
}

// Event listeners
sendBtn.addEventListener("click", sendMessage);
input.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

// 👋 First welcome message when chatbot loads
window.onload = () => {
  addMessage("Hello, how can I help you?", "left");
};
