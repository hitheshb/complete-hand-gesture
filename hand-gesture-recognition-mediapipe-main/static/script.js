document.addEventListener("DOMContentLoaded", function () {
    // Mobile menu toggle
    const hamburger = document.querySelector(".hamburger");
    const navLinksContainer = document.querySelector(".nav-links");
  
    hamburger.addEventListener("click", function () {
      navLinksContainer.classList.toggle("active");
    });
  
    // Feature cards animation on hover
    const featureCards = document.querySelectorAll(".feature-card");
  
    featureCards.forEach((card) => {
      card.addEventListener("mouseover", function () {
        this.style.backgroundColor = "#f8f9fa";
      });
  
      card.addEventListener("mouseout", function () {
        this.style.backgroundColor = "#ecf0f1";
      });
    });
  
    // Handle conversion page functionality
    if (document.getElementById("gesture-to-text-page")) {
      document.querySelector(".btn").addEventListener("click", function() {
        alert("Camera functionality would be implemented here.");
      });
    }
  
    if (document.getElementById("gesture-to-speech-page")) {
      document.querySelector(".btn").addEventListener("click", function() {
        alert("Detection functionality would be implemented here.");
      });
    }
  
    if (document.getElementById("speech-to-gesture-page")) {
      document.querySelector(".btn").addEventListener("click", function() {
        alert("Microphone functionality would be implemented here.");
      });
    }
  
    // Close mobile menu when clicking anywhere on the page
    document.addEventListener("click", function(event) {
      if (
        !event.target.closest(".hamburger") &&
        !event.target.closest(".nav-links") &&
        navLinksContainer.classList.contains("active")
      ) {
        navLinksContainer.classList.remove("active");
      }
    });
  });