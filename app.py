<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MarketMentor - Smart Stock Market Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon@3.0.4/build/global/luxon.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <style>
        :root {
            --primary: #0a1930;
            --secondary: #1e3a8a;
            --accent: #3b82f6;
            --light: #1e293b;
            --dark: #f8fafc;
            --danger: #ef4444;
            --warning: #f59e0b;
            --success: #10b981;
            --sidebar-width: 280px;
            --header-height: 70px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background-color: #0f172a;
            color: #cbd5e1;
            overflow-x: hidden;
        }

        /* Sidebar Styles */
        .sidebar {
            position: fixed;
            top: 0;
            left: 0;
            height: 100%;
            width: var(--sidebar-width);
            background: linear-gradient(180deg, var(--primary) 0%, #0a1930 100%);
            color: white;
            z-index: 1000;
            overflow-y: auto;
            transition: all 0.3s ease;
            box-shadow: 3px 0 15px rgba(0, 0, 0, 0.1);
        }

        .sidebar-header {
            padding: 20px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .sidebar-header h3 {
            font-weight: 700;
            font-size: 1.5rem;
            margin: 0;
        }

        .sidebar-menu {
            list-style: none;
            padding: 10px 0;
        }

        .sidebar-menu li {
            margin-bottom: 5px;
        }

        .sidebar-menu a {
            display: flex;
            align-items: center;
            padding: 12px 20px;
            color: rgba(255, 255, 255, 0.9);
            text-decoration: none;
            transition: all 0.3s;
            border-left: 4px solid transparent;
        }

        .sidebar-menu a:hover, .sidebar-menu a.active {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            border-left: 4px solid var(--accent);
        }

        .sidebar-menu i {
            margin-right: 12px;
            font-size: 1.1rem;
            width: 24px;
            text-align: center;
        }

        /* Main Content */
        .main-content {
            margin-left: var(--sidebar-width);
            padding: 20px;
            padding-top: calc(var(--header-height) + 20px);
            min-height: 100vh;
        }

        /* Header */
        .header {
            position: fixed;
            top: 0;
            left: var(--sidebar-width);
            right: 0;
            height: var(--header-height);
            background: var(--primary);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            z-index: 900;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
        }

        .search-bar {
            width: 300px;
            position: relative;
        }

        .search-bar input {
            width: 100%;
            padding: 10px 15px;
            border-radius: 25px;
            border: 1px solid #334155;
            padding-left: 40px;
            background: #1e293b;
            color: #e2e8f0;
        }

        .search-bar i {
            position: absolute;
            left: 15px;
            top: 50%;
            transform: translateY(-50%);
            color: #94a3b8;
        }

        .user-actions {
            display: flex;
            align-items: center;
        }

        .notification-bell {
            position: relative;
            margin-right: 20px;
            cursor: pointer;
        }

        .notification-badge {
            position: absolute;
            top: -5px;
            right: -5px;
            background: var(--danger);
            color: white;
            border-radius: 50%;
            width: 18px;
            height: 18px;
            font-size: 0.7rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .user-profile {
            display: flex;
            align-items: center;
            cursor: pointer;
        }

        .user-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 10px;
        }

        /* Dashboard Cards */
        .dashboard-header {
            margin-bottom: 25px;
        }

        .dashboard-header h1 {
            font-weight: 700;
            color: var(--dark);
            margin-bottom: 5px;
        }

        .breadcrumb {
            background: transparent;
            padding: 0;
            margin-bottom: 0;
            color: #94a3b8;
        }

        .breadcrumb-item.active {
            color: var(--accent);
        }

        .card {
            border: none;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            margin-bottom: 24px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            background: #1e293b;
        }

        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }

        .card-header {
            background: #1e293b;
            border-bottom: 1px solid #334155;
            padding: 15px 20px;
            border-radius: 12px 12px 0 0 !important;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #e2e8f0;
        }

        .card-header i {
            color: var(--accent);
            margin-right: 8px;
        }

        .card-body {
            padding: 20px;
            color: #cbd5e1;
        }

        /* Index Cards */
        .index-card {
            text-align: center;
            padding: 15px;
            background: #1e293b;
            border-radius: 10px;
            margin-bottom: 15px;
            transition: all 0.3s ease;
        }

        .index-card:hover {
            background: #2d3748;
        }

        .index-value {
            font-size: 1.8rem;
            font-weight: 700;
            margin: 10px 0;
            color: #e2e8f0;
        }

        .index-change {
            font-weight: 600;
            font-size: 0.9rem;
        }

        .positive {
            color: var(--success);
        }

        .negative {
            color: var(--danger);
        }

        /* Tabs */
        .nav-tabs {
            border-bottom: 1px solid #334155;
        }

        .nav-tabs .nav-link {
            border: none;
            color: #64748b;
            font-weight: 500;
            padding: 10px 15px;
        }

        .nav-tabs .nav-link.active {
            color: var(--accent);
            border-bottom: 3px solid var(--accent);
            background: transparent;
        }

        /* Tables */
        .data-table {
            width: 100%;
            color: #cbd5e1;
        }

        .data-table th {
            background: #1e293b;
            font-weight: 600;
            padding: 12px 15px;
            color: #e2e8f0;
            border-top: 1px solid #334155;
        }

        .data-table td {
            padding: 12px 15px;
            border-top: 1px solid #334155;
        }

        .data-table tr:hover {
            background: #2d3748;
        }

        /* Charts */
        .chart-container {
            position: relative;
            height: 300px;
            width: 100%;
        }

        /* Buttons */
        .btn-primary {
            background: var(--accent);
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            color: white;
        }

        .btn-primary:hover {
            background: #2563eb;
        }

        .btn-outline-primary {
            border-color: var(--accent);
            color: var(--accent);
        }

        .btn-outline-primary:hover {
            background: var(--accent);
            color: white;
        }

        /* Custom Toggle */
        .toggle-switch {
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }

        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }

        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #475569;
            transition: .4s;
            border-radius: 24px;
        }

        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }

        input:checked + .slider {
            background-color: var(--accent);
        }

        input:checked + .slider:before {
            transform: translateX(26px);
        }

        /* Footer */
        .footer {
            background: var(--primary);
            padding: 20px;
            margin-left: var(--sidebar-width);
            text-align: center;
            border-top: 1px solid #334155;
            color: #64748b;
        }

        /* Page Content */
        .page-content {
            display: none;
        }

        .page-content.active {
            display: block;
            animation: fadeIn 0.5s;
        }

        /* Chatbot */
        .chatbot-container {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }

        .chatbot-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        }

        .chatbot-window {
            position: absolute;
            bottom: 70px;
            right: 0;
            width: 350px;
            height: 450px;
            background: #1e293b;
            border-radius: 12px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.2);
            display: none;
            flex-direction: column;
            overflow: hidden;
            border: 1px solid #334155;
        }

        .chatbot-header {
            padding: 15px;
            background: var(--primary);
            color: white;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .chatbot-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }

        .message {
            max-width: 80%;
            padding: 10px 15px;
            margin-bottom: 10px;
            border-radius: 18px;
            line-height: 1.4;
        }

        .bot-message {
            background: #334155;
            color: #e2e8f0;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }

        .user-message {
            background: var(--accent);
            color: white;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }

        .chatbot-input {
            display: flex;
            padding: 10px;
            border-top: 1px solid #334155;
            background: #1e293b;
        }

        .chatbot-input input {
            flex: 1;
            padding: 10px 15px;
            border: 1px solid #334155;
            border-radius: 20px;
            background: #0f172a;
            color: #e2e8f0;
            outline: none;
        }

        .chatbot-input button {
            margin-left: 10px;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 10px 15px;
            cursor: pointer;
        }

        /* Responsive */
        @media (max-width: 992px) {
            .sidebar {
                width: 70px;
                overflow: visible;
            }
            
            .sidebar .sidebar-header h3,
            .sidebar .menu-text {
                display: none;
            }
            
            .sidebar .sidebar-menu a {
                justify-content: center;
                padding: 15px;
            }
            
            .sidebar .sidebar-menu i {
                margin-right: 0;
                font-size: 1.3rem;
            }
            
            .main-content, .header, .footer {
                margin-left: 70px;
            }
            
            .search-bar {
                width: 200px;
            }
        }

        @media (max-width: 768px) {
            .sidebar {
                left: -70px;
            }
            
            .main-content, .header, .footer {
                margin-left: 0;
            }
            
            .header {
                left: 0;
            }
            
            .menu-toggle {
                display: block !important;
            }
        }

        .menu-toggle {
            display: none;
            background: var(--accent);
            color: white;
            border: none;
            width: 40px;
            height: 40px;
            border-radius: 50%;
            font-size: 1.2rem;
            cursor: pointer;
            margin-right: 15px;
        }

        /* Custom animations */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }

        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #1e293b;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--accent);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #2563eb;
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h3>MarketMentor</h3>
        </div>
        <ul class="sidebar-menu">
            <li><a href="#" class="active" data-page="home"><i class="fas fa-home"></i> <span class="menu-text">Home</span></a></li>
            <li><a href="#" data-page="company"><i class="fas fa-building"></i> <span class="menu-text">Company Overview</span></a></li>
            <li><a href="#" data-page="movers"><i class="fas fa-chart-line"></i> <span class="menu-text">Market Movers</span></a></li>
            <li><a href="#" data-page="fnO"><i class="fas fa-exchange-alt"></i> <span class="menu-text">F&O</span></a></li>
            <li><a href="#" data-page="global"><i class="fas fa-globe"></i> <span class="menu-text">Global Markets</span></a></li>
            <li><a href="#" data-page="mutual"><i class="fas fa-chart-pie"></i> <span class="menu-text">Mutual Funds</span></a></li>
            <li><a href="#" data-page="sip"><i class="fas fa-calculator"></i> <span class="menu-text">SIP Calculator</span></a></li>
            <li><a href="#" data-page="ipo"><i class="fas fa-file-invoice-dollar"></i> <span class="menu-text">IPO Tracker</span></a></li>
            <li><a href="#" data-page="predictions"><i class="fas fa-crystal-ball"></i> <span class="menu-text">Predictions</span></a></li>
            <li><a href="#" data-page="nav"><i class="fas fa-money-bill-wave"></i> <span class="menu-text">NAV Viewer</span></a></li>
            <li><a href="#" data-page="sectors"><i class="fas fa-industry"></i> <span class="menu-text">Sectors</span></a></li>
            <li><a href="#" data-page="news"><i class="fas fa-newspaper"></i> <span class="menu-text">News</span></a></li>
            <li><a href="#" data-page="learning"><i class="fas fa-graduation-cap"></i> <span class="menu-text">Learning</span></a></li>
            <li><a href="#" data-page="volume"><i class="fas fa-wave-square"></i> <span class="menu-text">Volume Spike</span></a></li>
            <li><a href="#" data-page="screener"><i class="fas fa-filter"></i> <span class="menu-text">Stock Screener</span></a></li>
            <li><a href="#" data-page="buysell"><i class="fas fa-hand-holding-usd"></i> <span class="menu-text">Buy/Sell Predictor</span></a></li>
            <li><a href="#" data-page="sentiment"><i class="fas fa-smile"></i> <span class="menu-text">News Sentiment</span></a></li>
        </ul>
    </div>

    <!-- Header -->
    <div class="header">
        <div class="d-flex align-items-center">
            <button class="menu-toggle"><i class="fas fa-bars"></i></button>
            <div class="search-bar">
                <i class="fas fa-search"></i>
                <input type="text" id="searchInput" placeholder="Search stocks, indices, news...">
            </div>
        </div>
        <div class="user-actions">
            <div class="notification-bell">
                <i class="fas fa-bell"></i>
                <span class="notification-badge">3</span>
            </div>
            <div class="user-profile">
                <div class="user-avatar">AB</div>
                <div class="user-info d-none d-md-block">
                    <div class="user-name">Ashwik Bire</div>
                    <div class="user-role" style="color: #94a3b8;">Investor</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-content">
        <!-- Home Page -->
        <div id="home-page" class="page-content active">
            <div class="dashboard-header">
                <h1>Market Overview</h1>
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="#">Home</a></li>
                        <li class="breadcrumb-item active" aria-current="page">Dashboard</li>
                    </ol>
                </nav>
            </div>

            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-chart-bar"></i> Major Indices Performance</span>
                            <div class="dropdown">
                                <button class="btn btn-sm btn-outline-primary dropdown-toggle" type="button" id="timeRangeDropdown" data-bs-toggle="dropdown" aria-expanded="false">
                                    1D
                                </button>
                                <ul class="dropdown-menu" aria-labelledby="timeRangeDropdown">
                                    <li><a class="dropdown-item" href="#">1D</a></li>
                                    <li><a class="dropdown-item" href="#">1W</a></li>
                                    <li><a class="dropdown-item" href="#">1M</a></li>
                                    <li><a class="dropdown-item" href="#">3M</a></li>
                                    <li><a class="dropdown-item" href="#">1Y</a></li>
                                </ul>
                            </div>
                        </div>
                        <div class="card-body">
                            <div class="row" id="indices-container">
                                <!-- Indices will be populated here by JavaScript -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-chart-line"></i> Portfolio Performance</span>
                        </div>
                        <div class="card-body">
                            <div class="chart-container">
                                <canvas id="portfolioChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-list"></i> Watchlist</span>
                            <button class="btn btn-sm btn-primary"><i class="fas fa-plus"></i> Add</button>
                        </div>
                        <div class="card-body">
                            <table class="data-table">
                                <thead>
                                    <tr>
                                        <th>Symbol</th>
                                        <th>Price</th>
                                        <th>Change</th>
                                    </tr>
                                </thead>
                                <tbody id="watchlist-container">
                                    <!-- Watchlist will be populated here by JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </div>

            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-newspaper"></i> Market News</span>
                            <a href="#" class="btn btn-sm btn-outline-primary">View All</a>
                        </div>
                        <div class="card-body">
                            <div id="news-container">
                                <!-- News will be populated here by JavaScript -->
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-industry"></i> Top Sectors</span>
                        </div>
                        <div class="card-body">
                            <canvas id="sectorsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Other Pages (initially hidden) -->
        <div id="company-page" class="page-content">
            <div class="dashboard-header">
                <h1>Company Overview</h1>
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="#">Home</a></li>
                        <li class="breadcrumb-item active" aria-current="page">Company Overview</li>
                    </ol>
                </nav>
            </div>
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-building"></i> Company Search</span>
                </div>
                <div class="card-body">
                    <div class="row mb-4">
                        <div class="col-md-6">
                            <div class="search-bar">
                                <i class="fas fa-search"></i>
                                <input type="text" id="companySearch" placeholder="Search for a company...">
                            </div>
                        </div>
                        <div class="col-md-6">
                            <button class="btn btn-primary" id="searchCompanyBtn">Search</button>
                        </div>
                    </div>
                    <div id="company-details">
                        <p class="text-center">Search for a company to view details</p>
                    </div>
                </div>
            </div>
        </div>

        <div id="sip-page" class="page-content">
            <div class="dashboard-header">
                <h1>SIP Calculator</h1>
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="#">Home</a></li>
                        <li class="breadcrumb-item active" aria-current="page">SIP Calculator</li>
                    </ol>
                </nav>
            </div>
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-calculator"></i> SIP Input</span>
                        </div>
                        <div class="card-body">
                            <form id="sipForm">
                                <div class="mb-3">
                                    <label for="monthlyInvestment" class="form-label">Monthly Investment (₹)</label>
                                    <input type="number" class="form-control" id="monthlyInvestment" value="5000" min="500" step="500">
                                </div>
                                <div class="mb-3">
                                    <label for="investmentPeriod" class="form-label">Investment Period (Years)</label>
                                    <input type="number" class="form-control" id="investmentPeriod" value="10" min="1" max="30">
                                </div>
                                <div class="mb-3">
                                    <label for="expectedReturn" class="form-label">Expected Annual Return (%)</label>
                                    <input type="number" class="form-control" id="expectedReturn" value="12" min="1" max="30" step="0.1">
                                </div>
                                <button type="button" class="btn btn-primary" id="calculateSip">Calculate</button>
                            </form>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <span><i class="fas fa-chart-pie"></i> SIP Results</span>
                        </div>
                        <div class="card-body">
                            <div id="sipResults">
                                <p class="text-center">Enter values and click Calculate to see results</p>
                            </div>
                            <div class="chart-container">
                                <canvas id="sipChart"></canvas>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Add other page content here -->
        <div id="news-page" class="page-content">
            <div class="dashboard-header">
                <h1>Financial News</h1>
                <nav aria-label="breadcrumb">
                    <ol class="breadcrumb">
                        <li class="breadcrumb-item"><a href="#">Home</a></li>
                        <li class="breadcrumb-item active" aria-current="page">News</li>
                    </ol>
                </nav>
            </div>
            <div class="card">
                <div class="card-header">
                    <span><i class="fas fa-newspaper"></i> Latest Market News</span>
                </div>
                <div class="card-body">
                    <div id="news-list">
                        <!-- News articles will be populated here by JavaScript -->
                    </div>
                </div>
            </div>
        </div>

        <!-- Add more pages for other menu items -->
    </div>

    <!-- Footer -->
    <div class="footer">
        <p>© 2023 MarketMentor. All rights reserved. | Data provided by financial APIs</p>
    </div>

    <!-- Chatbot -->
    <div class="chatbot-container">
        <div class="chatbot-btn" id="chatbotToggle">
            <i class="fas fa-comment-dots"></i>
        </div>
        <div class="chatbot-window" id="chatbotWindow">
            <div class="chatbot-header">
                <span>MarketMentor Assistant</span>
                <i class="fas fa-times" id="closeChatbot"></i>
            </div>
            <div class="chatbot-messages" id="chatbotMessages">
                <div class="message bot-message">
                    Hello! I'm your MarketMentor assistant. How can I help you today?
                </div>
            </div>
            <div class="chatbot-input">
                <input type="text" id="chatbotInput" placeholder="Type your message...">
                <button id="sendMessage"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Page Navigation
        document.querySelectorAll('.sidebar-menu a').forEach(item => {
            item.addEventListener('click', function(e) {
                e.preventDefault();
                const pageId = this.getAttribute('data-page');
                
                // Update active menu item
                document.querySelectorAll('.sidebar-menu a').forEach(link => {
                    link.classList.remove('active');
                });
                this.classList.add('active');
                
                // Show the corresponding page
                document.querySelectorAll('.page-content').forEach(page => {
                    page.classList.remove('active');
                });
                document.getElementById(`${pageId}-page`).classList.add('active');
            });
        });

        // Toggle sidebar on mobile
        document.querySelector('.menu-toggle').addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('show');
        });

        // Chatbot functionality
        document.getElementById('chatbotToggle').addEventListener('click', function() {
            document.getElementById('chatbotWindow').style.display = 'flex';
        });

        document.getElementById('closeChatbot').addEventListener('click', function() {
            document.getElementById('chatbotWindow').style.display = 'none';
        });

        document.getElementById('sendMessage').addEventListener('click', function() {
            const input = document.getElementById('chatbotInput');
            const message = input.value.trim();
            
            if (message) {
                // Add user message
                const userMessageElement = document.createElement('div');
                userMessageElement.classList.add('message', 'user-message');
                userMessageElement.textContent = message;
                document.getElementById('chatbotMessages').appendChild(userMessageElement);
                
                // Clear input
                input.value = '';
                
                // Scroll to bottom
                document.getElementById('chatbotMessages').scrollTop = document.getElementById('chatbotMessages').scrollHeight;
                
                // Simulate bot response
                setTimeout(() => {
                    const botResponse = generateBotResponse(message);
                    const botMessageElement = document.createElement('div');
                    botMessageElement.classList.add('message', 'bot-message');
                    botMessageElement.textContent = botResponse;
                    document.getElementById('chatbotMessages').appendChild(botMessageElement);
                    
                    // Scroll to bottom
                    document.getElementById('chatbotMessages').scrollTop = document.getElementById('chatbotMessages').scrollHeight;
                }, 1000);
            }
        });

        // Allow sending message with Enter key
        document.getElementById('chatbotInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                document.getElementById('sendMessage').click();
            }
        });

        // Generate bot response based on user input
        function generateBotResponse(message) {
            const lowerMessage = message.toLowerCase();
            
            if (lowerMessage.includes('hello') || lowerMessage.includes('hi') || lowerMessage.includes('hey')) {
                return "Hello! How can I assist you with your investment journey today?";
            } else if (lowerMessage.includes('stock') || lowerMessage.includes('share')) {
                return "I can help you with stock information. Which company are you interested in?";
            } else if (lowerMessage.includes('sip') || lowerMessage.includes('investment')) {
                return "Systematic Investment Plans (SIPs) are a great way to invest in mutual funds regularly. You can use our SIP calculator to estimate returns.";
            } else if (lowerMessage.includes('market') || lowerMessage.includes('nifty') || lowerMessage.includes('sensex')) {
                return "Currently, the market is showing mixed signals. For detailed analysis, check out the Market Overview section.";
            } else if (lowerMessage.includes('thank')) {
                return "You're welcome! Feel free to ask if you have more questions.";
            } else {
                return "I'm still learning about financial markets. Could you please rephrase your question or ask about stocks, SIP, or market trends?";
            }
        }

        // Sample data for indices (in a real app, this would come from an API)
        const indicesData = [
            { name: 'NIFTY 50', value: 19632.55, change: 124.65, changePercent: 0.64 },
            { name: 'SENSEX', value: 65958.21, change: 382.93, changePercent: 0.58 },
            { name: 'NIFTY BANK', value: 44872.90, change: -132.45, changePercent: -0.29 },
            { name: 'NIFTY IT', value: 34125.65, change: 523.12, changePercent: 1.56 }
        ];

        // Populate indices
        const indicesContainer = document.getElementById('indices-container');
        indicesData.forEach(index => {
            const col = document.createElement('div');
            col.className = 'col-md-3 col-sm-6';
            
            const isPositive = index.change >= 0;
            const changeClass = isPositive ? 'positive' : 'negative';
            const changeIcon = isPositive ? '▲' : '▼';
            
            col.innerHTML = `
                <div class="index-card">
                    <h5>${index.name}</h5>
                    <div class="index-value">${index.value.toLocaleString()}</div>
                    <div class="index-change ${changeClass}">
                        ${changeIcon} ${index.change.toLocaleString()} (${index.changePercent}%)
                    </div>
                </div>
            `;
            
            indicesContainer.appendChild(col);
        });

        // Sample watchlist data
        const watchlistData = [
            { symbol: 'RELIANCE', price: 2587.45, change: 1.2 },
            { symbol: 'HDFC', price: 1642.30, change: -0.8 },
            { symbol: 'INFY', price: 1485.60, change: 2.1 },
            { symbol: 'TCS', price: 3325.75, change: 0.7 },
            { symbol: 'HUL', price: 2436.90, change: -0.5 }
        ];

        // Populate watchlist
        const watchlistContainer = document.getElementById('watchlist-container');
        watchlistData.forEach(stock => {
            const row = document.createElement('tr');
            const isPositive = stock.change >= 0;
            const changeClass = isPositive ? 'positive' : 'negative';
            const changeIcon = isPositive ? '▲' : '▼';
            
            row.innerHTML = `
                <td>${stock.symbol}</td>
                <td>₹${stock.price.toLocaleString()}</td>
                <td class="${changeClass}">${changeIcon} ${Math.abs(stock.change)}%</td>
            `;
            
            watchlistContainer.appendChild(row);
        });

        // Sample news data
        const newsData = [
            { title: 'RBI Keeps Repo Rate Unchanged at 6.5%', source: 'Economic Times', time: '2 hours ago' },
            { title: 'Infosys Reports Better-Than-Expected Q2 Results', source: 'Business Standard', time: '4 hours ago' },
            { title: 'Government Plans New Policy to Boost Manufacturing', source: 'MoneyControl', time: '6 hours ago' }
        ];

        // Populate news
        const newsContainer = document.getElementById('news-container');
        newsData.forEach(news => {
            const newsItem = document.createElement('div');
            newsItem.className = 'mb-3';
            newsItem.innerHTML = `
                <h6>${news.title}</h6>
                <div class="d-flex justify-content-between text-muted">
                    <span>${news.source}</span>
                    <span>${news.time}</span>
                </div>
                <hr>
            `;
            newsContainer.appendChild(newsItem);
        });

        // Initialize charts
        // Portfolio chart
        const portfolioCtx = document.getElementById('portfolioChart').getContext('2d');
        const portfolioChart = new Chart(portfolioCtx, {
            type: 'line',
            data: {
                labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct'],
                datasets: [{
                    label: 'Portfolio Value (₹)',
                    data: [500000, 520000, 510000, 535000, 545000, 560000, 580000, 575000, 590000, 610000],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        }
                    },
                    x: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#cbd5e1'
                        }
                    }
                }
            }
        });

        // Sectors chart
        const sectorsCtx = document.getElementById('sectorsChart').getContext('2d');
        const sectorsChart = new Chart(sectorsCtx, {
            type: 'doughnut',
            data: {
                labels: ['IT', 'Finance', 'Healthcare', 'Auto', 'FMCG', 'Energy'],
                datasets: [{
                    data: [25, 22, 18, 15, 12, 8],
                    backgroundColor: [
                        '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'
                    ]
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#cbd5e1'
                        }
                    }
                }
            }
        });

        // SIP Calculator functionality
        document.getElementById('calculateSip').addEventListener('click', function() {
            const monthlyInvestment = parseFloat(document.getElementById('monthlyInvestment').value);
            const investmentPeriod = parseInt(document.getElementById('investmentPeriod').value);
            const expectedReturn = parseFloat(document.getElementById('expectedReturn').value);
            
            // Calculate SIP
            const monthlyRate = expectedReturn / 100 / 12;
            const months = investmentPeriod * 12;
            const futureValue = monthlyInvestment * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate) * (1 + monthlyRate);
            const totalInvestment = monthlyInvestment * months;
            const totalGains = futureValue - totalInvestment;
            
            // Display results
            document.getElementById('sipResults').innerHTML = `
                <div class="row text-center">
                    <div class="col-md-4 mb-3">
                        <div class="fw-bold">Invested Amount</div>
                        <div class="fs-4">₹${totalInvestment.toLocaleString()}</div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="fw-bold">Est. Returns</div>
                        <div class="fs-4 text-success">₹${totalGains.toLocaleString()}</div>
                    </div>
                    <div class="col-md-4 mb-3">
                        <div class="fw-bold">Total Value</div>
                        <div class="fs-4 text-primary">₹${futureValue.toLocaleString()}</div>
                    </div>
                </div>
            `;
            
            // Update SIP chart
            updateSipChart(totalInvestment, totalGains);
        });

        function updateSipChart(invested, returns) {
            const sipCtx = document.getElementById('sipChart').getContext('2d');
            
            if (window.sipChartInstance) {
                window.sipChartInstance.destroy();
            }
            
            window.sipChartInstance = new Chart(sipCtx, {
                type: 'pie',
                data: {
                    labels: ['Amount Invested', 'Estimated Returns'],
                    datasets: [{
                        data: [invested, returns],
                        backgroundColor: ['#3b82f6', '#10b981']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom',
                            labels: {
                                color: '#cbd5e1'
                            }
                        }
                    }
                }
            });
        }

        // Load news using News API
        function loadNews() {
            const apiKey = '0b08be107dca45d3be30ca7e06544408';
            const url = `https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey=${apiKey}`;
            
            fetch(url)
                .then(response => response.json())
                .then(data => {
                    const newsList = document.getElementById('news-list');
                    newsList.innerHTML = '';
                    
                    if (data.articles && data.articles.length > 0) {
                        data.articles.slice(0, 10).forEach(article => {
                            const newsItem = document.createElement('div');
                            newsItem.className = 'mb-4';
                            newsItem.innerHTML = `
                                <h5><a href="${article.url}" target="_blank" style="color: #3b82f6; text-decoration: none;">${article.title}</a></h5>
                                <p>${article.description || ''}</p>
                                <div class="d-flex justify-content-between text-muted">
                                    <span>${article.source.name}</span>
                                    <span>${new Date(article.publishedAt).toLocaleDateString()}</span>
                                </div>
                                <hr>
                            `;
                            newsList.appendChild(newsItem);
                        });
                    } else {
                        newsList.innerHTML = '<p class="text-center">No news available at the moment.</p>';
                    }
                })
                .catch(error => {
                    console.error('Error fetching news:', error);
                    document.getElementById('news-list').innerHTML = '<p class="text-center">Failed to load news. Please try again later.</p>';
                });
        }

        // Load news when news page is accessed
        document.querySelector('[data-page="news"]').addEventListener('click', loadNews);

        // Company search functionality
        document.getElementById('searchCompanyBtn').addEventListener('click', function() {
            const companyName = document.getElementById('companySearch').value.trim();
            
            if (companyName) {
                // In a real app, this would fetch data from an API
                // For demo purposes, we'll use mock data
                const companyDetails = document.getElementById('company-details');
                companyDetails.innerHTML = `
                    <div class="text-center">
                        <div class="spinner-border text-primary" role="status">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <p>Loading company data...</p>
                    </div>
                `;
                
                setTimeout(() => {
                    companyDetails.innerHTML = `
                        <div class="company-header mb-4">
                            <h3>${companyName.toUpperCase()} Company Limited</h3>
                            <p>NSE: ${companyName.substring(0, 4).toUpperCase()} | BSE: 543210</p>
                        </div>
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h5 class="card-title">Current Price</h5>
                                        <h2 class="text-primary">₹${(Math.random() * 5000 + 100).toFixed(2)}</h2>
                                        <p class="${Math.random() > 0.5 ? 'text-success' : 'text-danger'}">
                                            ${Math.random() > 0.5 ? '▲' : '▼'} ${(Math.random() * 10).toFixed(2)}%
                                        </p>
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card mb-3">
                                    <div class="card-body">
                                        <h5 class="card-title">Key Metrics</h5>
                                        <div class="row">
                                            <div class="col-6">
                                                <p>Market Cap</p>
                                                <p>P/E Ratio</p>
                                                <p>Dividend Yield</p>
                                            </div>
                                            <div class="col-6">
                                                <p>₹${(Math.random() * 100000).toFixed(2)} Cr</p>
                                                <p>${(Math.random() * 50 + 10).toFixed(2)}</p>
                                                <p>${(Math.random() * 5).toFixed(2)}%</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">About Company</h5>
                                <p>${companyName} is a leading company in its sector with a strong market presence and consistent financial performance. The company has shown resilience in challenging market conditions and continues to innovate in its product offerings.</p>
                                <h5 class="card-title mt-4">Recent Performance</h5>
                                <canvas id="companyChart" height="150"></canvas>
                            </div>
                        </div>
                    `;
                    
                    // Initialize company chart
                    const companyCtx = document.getElementById('companyChart').getContext('2d');
                    new Chart(companyCtx, {
                        type: 'line',
                        data: {
                            labels: ['Q1', 'Q2', 'Q3', 'Q4'],
                            datasets: [{
                                label: 'Revenue (₹ Cr)',
                                data: [
                                    Math.random() * 1000 + 500,
                                    Math.random() * 1000 + 600,
                                    Math.random() * 1000 + 700,
                                    Math.random() * 1000 + 800
                                ],
                                borderColor: '#3b82f6',
                                tension: 0.3
                            }]
                        },
                        options: {
                            scales: {
                                y: {
                                    grid: {
                                        color: 'rgba(255, 255, 255, 0.1)'
                                    },
                                    ticks: {
                                        color: '#cbd5e1'
                                    }
                                },
                                x: {
                                    grid: {
                                        color: 'rgba(255, 255, 255, 0.1)'
                                    },
                                    ticks: {
                                        color: '#cbd5e1'
                                    }
                                }
                            }
                        }
                    });
                }, 1500);
            }
        });
    </script>
</body>
</html>
