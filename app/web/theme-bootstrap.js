// SPDX-License-Identifier: Apache-2.0
(function() {
  try {
    var theme = localStorage.getItem("knight-theme") || "dark";
    document.documentElement.setAttribute("data-theme", theme);
  } catch (_) {}
})();
