const sendBtn = document.getElementById('send');
const output  = document.getElementById('output');
const costDiv = document.getElementById('cost');

sendBtn.addEventListener('click', async () => {
  const message = document.getElementById('message').value.trim();
  const model   = document.getElementById('model').value;
  if (!message) return;

  output.textContent = '';
  costDiv.textContent = '';
  sendBtn.disabled = true;

  const response = await fetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, model }),
  });

  const reader  = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Split off the cost summary line if it has arrived
    const costMatch = buffer.match(/^([\s\S]*?)(\n\n\[.+\])$/);
    if (costMatch) {
      output.textContent = costMatch[1];
      costDiv.textContent = costMatch[2].trim();
    } else {
      output.textContent = buffer;
    }
  }

  sendBtn.disabled = false;
});
