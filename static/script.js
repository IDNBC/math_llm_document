document.getElementById('toggleSidebar').onclick = () => {
    document.getElementById('sidebar').classList.toggle('collapsed');
};

function loadChat(title) {
    document.getElementById('chatDisplay').innerHTML = `<h2>${title}</h2><p>チャット内容をここに表示</p>`;
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    if (message) {
        const display = document.getElementById('chatDisplay');
        const userMessage = document.createElement('p');
        userMessage.textContent = `ユーザー: ${message}`;
        display.appendChild(userMessage);
        input.value = '';

        // FlaskバックエンドからLaTeX数式を取得
        const response = await fetch('/latex');  // '/latex'エンドポイントにGETリクエストを送信
        if (response.ok) {
            const data = await response.json();
            const aiMessage = document.createElement('p');
            aiMessage.innerHTML = `AI: $${data.latex}$`;  // MathJaxでレンダリングするためにドル記号で囲む
            display.appendChild(aiMessage);

            MathJax.typesetPromise();  // MathJaxにDOMを再描画させる
        } else {
            const aiMessage = document.createElement('p');
            aiMessage.textContent = 'エラーが発生しました。';
            display.appendChild(aiMessage);
        }
    }
}