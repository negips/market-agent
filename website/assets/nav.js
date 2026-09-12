(() => {
  const NAV_ITEMS = [
    {
      href:  'setup.html',
      label: 'Setup',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="3"/>
                <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/>
              </svg>`,
    },
    {
      href:  'companies.html',
      label: 'NSE Companies',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <rect x="3" y="3" width="18" height="18" rx="2"/>
                <path d="M3 9h18M3 15h18M9 3v18"/>
              </svg>`,
    },
    {
      href:  'watchlist.html',
      label: 'Watchlist',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <rect x="3" y="4" width="18" height="18" rx="2"/>
                <path d="M16 2v4M8 2v4M3 10h18"/>
                <path d="M8 14h.01M12 14h.01M16 14h.01M8 18h.01M12 18h.01M16 18h.01"/>
              </svg>`,
    },
    {
      href:  'training.html',
      label: 'Training',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
              </svg>`,
    },
    {
      href:  'predict.html',
      label: 'Predict',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <circle cx="11" cy="11" r="8"/>
                <path d="M21 21l-4.35-4.35M11 8v6M8 11h6"/>
              </svg>`,
    },
    {
      href:  'architecture.html',
      label: 'Architecture',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <rect x="2" y="3" width="6" height="5" rx="1"/>
                <rect x="9" y="3" width="6" height="5" rx="1"/>
                <rect x="16" y="3" width="6" height="5" rx="1"/>
                <rect x="5" y="13" width="6" height="5" rx="1"/>
                <rect x="13" y="13" width="6" height="5" rx="1"/>
                <path d="M5 8v2.5a1 1 0 001 1h12a1 1 0 001-1V8"/>
                <line x1="11" y1="11.5" x2="11" y2="13"/>
                <line x1="16" y1="11.5" x2="16" y2="13"/>
              </svg>`,
    },
    {
      href:  'jumps.html',
      label: 'Jumps',
      icon:  `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.6" viewBox="0 0 24 24">
                <polyline points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
              </svg>`,
    },
  ];

  function injectSidebar() {
    const current = window.location.pathname.split('/').pop() || 'index.html';

    const nav = document.createElement('nav');
    nav.className = 'sidebar';
    nav.innerHTML = `
      <div class="sidebar-logo">
        <span class="logo-name">market-agent</span>
        <span class="logo-sub">NSE / BSE · India</span>
      </div>
      <div class="nav-section">
        <span class="nav-section-label">Navigation</span>
        <ul class="nav-list">
          ${NAV_ITEMS.map(item => `
            <li>
              <a class="nav-link${current === item.href ? ' active' : ''}" href="${item.href}">
                <span class="nav-icon">${item.icon}</span>
                <span class="nav-label">${item.label}</span>
              </a>
            </li>
          `).join('')}
        </ul>
      </div>
      <div class="sidebar-footer">
        Served locally<br>
        <span id="nav-timestamp"></span>
      </div>
    `;

    document.body.prepend(nav);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', injectSidebar);
  } else {
    injectSidebar();
  }
})();
