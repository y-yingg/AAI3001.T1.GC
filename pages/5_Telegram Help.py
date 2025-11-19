import streamlit as st

def main():
    st.set_page_config(page_title="Telegram Help", page_icon="💬", layout="wide")

    # ---------- CSS ----------
    st.markdown("""
    <style>
      .section {
        background: rgba(240,240,240,0.1);
        border-radius: 14px;
        padding: 25px 32px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        margin-bottom: 25px;
        line-height: 1.7;
      }
      .title {
        color: #2d6cdf;
        font-weight: 800;
        font-size: 1.8rem;
        margin-bottom: 8px;
        text-align: center;
      }
      .subtitle {
        text-align: center;
        color: #bbb;
        margin-bottom: 30px;
        font-size: 1.05rem;
      }
      .section-title {
        color: #ffd966;
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 8px;
        border-left: 5px solid #2d6cdf;
        padding-left: 10px;
      }
      ul { margin-left: 1.2em; }
      li { margin-bottom: 0.5em; }
      code {
        background-color: rgba(0,0,0,0.25);
        border-radius: 5px;
        padding: 2px 6px;
        font-size: 0.95rem;
        color: #a8ff9f;
      }
      a { color:#79b8ff; text-decoration:none; }
      a:hover { text-decoration:underline; }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Content ----------
    st.markdown("<div class='title'>How to Get Your Telegram Bot Token & Chat ID</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle'>Quick reference based on "
        "<a href='https://gist.github.com/nafiesl/4ad622f344cd1dc3bb1ecbe468ff9f8a' target='_blank' rel='noopener noreferrer'>nafiesl’s guide</a>"
        "</div>",
        unsafe_allow_html=True
    )

    # Create Bot Token
    st.markdown("""
    <div class="section">
      <div class="section-title">1️⃣ Create a Telegram Bot and Get Your Bot Token</div>
      <ul>
        <li>Open the <strong>Telegram</strong> app and search for <code>@BotFather</code>.</li>
        <li>Click <strong>Start</strong>, then choose <strong>/newbot</strong>.</li>
        <li>Follow the prompts to name your bot — you’ll receive a message such as:<br>
            <em>"Done! Congratulations on your new bot …"</em></li>
        <li>Telegram will show a line that says:<br>
            <code>Use this token to access the HTTP API: 123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11</code></li>
        <li><strong>Copy and keep this token private!</strong> It allows control of your bot.</li>
      </ul>
      <p style="font-size:0.9rem;color:#aaa;">📘 Reference: <a href="https://core.telegram.org/bots/api" target="_blank">Telegram Bot API Docs</a></p>
    </div>
    """, unsafe_allow_html=True)

    # Private Chat
    st.markdown("""
    <div class="section">
      <div class="section-title">2️⃣ Get Chat ID for a Private Chat</div>
      <ul>
        <li>Search your newly created bot in Telegram and click <strong>Start</strong>.</li>
        <li>Then open this link in your browser (replace the token):<br>
            <code>https://api.telegram.org/bot&lt;your_token&gt;/getUpdates</code></li>
        <li>Look for the value of <code>result[0].message.chat.id</code> — that’s your Chat ID.</li>
        <li>Test it by visiting:<br>
            <code>https://api.telegram.org/bot&lt;your_token&gt;/sendMessage?chat_id=&lt;your_chat_id&gt;&amp;text=hello</code></li>
        <li>If the bot replies in your chat, it’s working correctly.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Channel Chat
    st.markdown("""
    <div class="section">
      <div class="section-title">3️⃣ Get Chat ID for a Channel</div>
      <ul>
        <li>Add your bot as a member of the channel.</li>
        <li>Send a message in that channel.</li>
        <li>Open the same <code>getUpdates</code> URL — the response will contain <code>channel_post.chat.id</code>.</li>
        <li>That value (usually starting with <code>-100</code>) is your channel Chat ID.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Group Chat
    st.markdown("""
    <div class="section">
      <div class="section-title">4️⃣ Get Chat ID for a Group Chat</div>
      <ul>
        <li>Open Telegram Desktop.</li>
        <li>Add your bot to the group and send any message.</li>
        <li>Right-click that message → <strong>Copy Message Link</strong>.</li>
        <li>You’ll see something like <code>https://t.me/c/194xxxx987/11</code> — the number <code>194xxxx987</code> is your group ID.</li>
        <li>Prefix it with <code>-100</code> when using it in the API:<br>
            <code>-100194xxxx987</code></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Group Topic
    st.markdown("""
    <div class="section">
      <div class="section-title">5️⃣ Get Chat ID for a Topic in a Group Chat</div>
      <ul>
        <li>Copy the full message link, e.g. <code>https://t.me/c/194xxxx987/11/13</code>.</li>
        <li>The middle number <code>11</code> is the topic ID (<code>message_thread_id</code>).</li>
        <li>Use it together with <code>chat_id</code> in your request:<br>
            <code>https://api.telegram.org/bot&lt;token&gt;/sendMessage?chat_id=-100194xxxx987&amp;message_thread_id=11&amp;text=test</code></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    # Tips / Troubleshooting
    st.markdown("""
    <div class="section">
      <div class="section-title">💡 Tips & Troubleshooting</div>
      <ul>
        <li>If <code>getUpdates</code> returns <code>"result": []</code>, send a message to your bot first, then refresh the URL.</li>
        <li>For group issues, disable “Group Privacy Mode” in @BotFather → My Bots → Bot Settings → Group Privacy → <strong>Disable</strong>.</li>
        <li>You can also use @RawDataBot to quickly view your chat IDs.</li>
        <li>Never share your Bot Token publicly - anyone can control your bot with it.</li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
