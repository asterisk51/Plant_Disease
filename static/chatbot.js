let history = [{ role: "system", content: "{{ system }}" }];
const chat = document.getElementById("chat");

function render() {
  chat.innerHTML = "";
  history.forEach(m => {
    let div = document.createElement("div");
    div.className = "msg " + (m.role === "user" ? "user" : m.role === "assistant" ? "assistant" : "sys");
    div.textContent = (m.role === "system" ? "System: " : "") + m.content;
    chat.appendChild(div);
  });
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage() {
  let text = document.getElementById("input").value.trim();
  if (!text) return;
  document.getElementById("input").value = "";
  history.push({ role: "user", content: text });
  render();
  try {
    let res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider: document.getElementById("provider").value,
        model: document.getElementById("model").value || null,
        message: text,
        history
      })
    });
    let data = await res.json();
    history.push({ role: "assistant", content: data.reply || ("Error: " + data.error) });
  } catch (e) {
    history.push({ role: "assistant", content: "Network error: " + e.message });
  }
  render();
}

render();
