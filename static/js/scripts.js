// Sidebar toggle functionality
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebar = document.getElementById("sidebar");
const backdrop = document.getElementById("backdrop");
const navbar = document.querySelector(".navbar");
const mainContent = document.getElementById("mainContent");

function toggleSidebar() {
  sidebar.classList.toggle("show");
  backdrop.classList.toggle("show");

  // Only apply sidebar-open class on larger screens
  if (window.innerWidth >= 992) {
    navbar.classList.toggle("sidebar-open");
    mainContent.classList.toggle("sidebar-open");
  }
}

function closeSidebar() {
  sidebar.classList.remove("show");
  backdrop.classList.remove("show");

  if (window.innerWidth >= 992) {
    navbar.classList.remove("sidebar-open");
    mainContent.classList.remove("sidebar-open");
  }
}

sidebarToggle.addEventListener("click", toggleSidebar);
backdrop.addEventListener("click", closeSidebar);

// Handle window resize
window.addEventListener("resize", function () {
  if (window.innerWidth >= 992) {
    // Desktop view - sidebar always visible
    sidebar.classList.remove("show");
    backdrop.classList.remove("show");
    navbar.classList.add("sidebar-open");
    mainContent.classList.add("sidebar-open");
  } else {
    // Mobile view - reset classes
    navbar.classList.remove("sidebar-open");
    mainContent.classList.remove("sidebar-open");
  }
});

// Initialize on page load
if (window.innerWidth >= 992) {
  navbar.classList.add("sidebar-open");
  mainContent.classList.add("sidebar-open");
}

// Navigation link active state
document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", function () {
    // Hapus kelas 'active' dari semua tautan
    document
      .querySelectorAll(".nav-link")
      .forEach((l) => l.classList.remove("active"));
    // Tambahkan kelas 'active' ke tautan yang diklik
    this.classList.add("active");
    // Tutup sidebar pada tampilan mobile
    if (window.innerWidth < 992) {
      closeSidebar();
    }
    // Navigasi akan dilakukan secara otomatis oleh browser ke href
  });
});

// Search functionality
const searchInput = document.getElementById("searchInput");
const searchClear = document.getElementById("searchClear");
const sidebarNav = document.getElementById("sidebarNav");
const noResults = document.getElementById("noResults");

function performSearch() {
  const searchTerm = searchInput.value.toLowerCase().trim();
  const navSections = sidebarNav.querySelectorAll(".nav-section");
  const navItems = sidebarNav.querySelectorAll(".nav-item");
  let hasResults = false;

  if (searchTerm === "") {
    // Show all items
    navSections.forEach((section) => section.classList.remove("hidden"));
    navItems.forEach((item) => item.classList.remove("hidden"));
    noResults.style.display = "none";
    searchClear.style.display = "none";
    return;
  }

  searchClear.style.display = "block";

  // Hide all sections first
  navSections.forEach((section) => section.classList.add("hidden"));
  navItems.forEach((item) => item.classList.add("hidden"));

  // Show matching sections and items
  navSections.forEach((section) => {
    const sectionTitle = section.querySelector(".nav-section-title");
    const sectionText = sectionTitle
      ? sectionTitle.textContent.toLowerCase()
      : "";
    const subItems = section.querySelectorAll(".nav-subsection .nav-item");
    let sectionHasMatch = false;

    // Check if section title matches
    if (sectionText.includes(searchTerm)) {
      section.classList.remove("hidden");
      subItems.forEach((item) => item.classList.remove("hidden"));
      sectionHasMatch = true;
      hasResults = true;
    } else {
      // Check individual sub-items
      subItems.forEach((item) => {
        const itemText = item.textContent.toLowerCase();
        if (itemText.includes(searchTerm)) {
          section.classList.remove("hidden");
          item.classList.remove("hidden");
          sectionHasMatch = true;
          hasResults = true;
        }
      });
    }
  });

  // Check main nav items (not in sections)
  const mainNavItems = sidebarNav.querySelectorAll(
    ".nav-item:not(.nav-section .nav-item)"
  );
  mainNavItems.forEach((item) => {
    const itemText = item.textContent.toLowerCase();
    if (itemText.includes(searchTerm)) {
      item.classList.remove("hidden");
      hasResults = true;
    }
  });

  // Show/hide no results message
  noResults.style.display = hasResults ? "none" : "block";
}

searchInput.addEventListener("input", performSearch);
searchInput.addEventListener("keyup", function (e) {
  if (e.key === "Escape") {
    this.value = "";
    performSearch();
    this.blur();
  }
});

searchClear.addEventListener("click", function () {
  searchInput.value = "";
  performSearch();
  searchInput.focus();
});
