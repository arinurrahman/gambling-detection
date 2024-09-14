document.addEventListener("DOMContentLoaded", (event) => {
  const resultElement = document.getElementById("result");
  if (resultElement && resultElement.textContent.trim() !== "") {
    resultElement.classList.add("visible");
    if (
      resultElement.textContent.includes(
        "Website ini kemungkinan adalah website judi."
      )
    ) {
      resultElement.classList.add("red");
    } else if (resultElement.textContent.includes("Gagal mengambil teks dari URL.")) {
      resultElement.classList.add("yellow");
    } else {
      resultElement.classList.add("green");
    }
  }
});
