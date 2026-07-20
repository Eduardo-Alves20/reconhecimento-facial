(function () {
  const toggle = document.getElementById("togglePassword");
  const input = document.getElementById("password");
  if (!toggle || !input) return;

  toggle.addEventListener("click", function () {
    const showing = input.type === "text";
    input.type = showing ? "password" : "text";
    toggle.setAttribute("aria-pressed", String(!showing));
    toggle.setAttribute("aria-label", showing ? "Mostrar senha" : "Ocultar senha");
    const eye = toggle.querySelector("[data-eye]");
    if (eye) eye.style.opacity = showing ? "1" : "0.5";
    input.focus();
  });
})();
