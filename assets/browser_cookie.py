"""
assets/browser_cookie.py - writes a cookie in the visitor's browser.

Why: Streamlit forgets st.session_state when the page is reloaded. To stay logged in,
app2.py keeps a random login token in a cookie. Python can READ cookies
(st.context.cookies, sent when the page loads) but cannot WRITE them, so this tiny
custom component (Streamlit Components v2) sets document.cookie from the page.

Usage:
    set_cookie("fa_login", token, max_age=7 * 24 * 3600)   # remember
    set_cookie("fa_login", "", max_age=0)                   # forget (logout)
"""

import streamlit as st

JS = """
export default function (component) {
  const { data } = component
  // "Secure" only on https (Streamlit Cloud); localhost is plain http
  const secure = location.protocol === "https:" ? "; Secure" : ""
  document.cookie = data.name + "=" + encodeURIComponent(data.value) +
                    "; path=/; max-age=" + data.max_age + "; SameSite=Lax" + secure
}
"""

_COOKIE = st.components.v2.component("browser_cookie", html="<span></span>", js=JS)


def set_cookie(name, value, max_age):
    """max_age in seconds; 0 deletes the cookie."""

    _COOKIE(key=f"cookie_{name}", data={"name": name, "value": value, "max_age": max_age})
