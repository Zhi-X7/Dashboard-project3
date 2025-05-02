# --- Standard Libraries ---
import os  # File and directory operations
import time  # Time tracking
import shutil  # File copying
import base64  # For encoding images to base64
import requests  # For making HTTP requests to weather API
from io import BytesIO  # In-memory byte buffer for image processing
from PIL import Image  # Image processing
import pandas as pd  # Data manipulation and analysis
from datetime import datetime, timedelta  # For handling dates and times

# --- Dash and Visualization Libraries ---
import dash  # Main Dash framework
import dash_bootstrap_components as dbc  # Bootstrap components for Dash
from dash import dcc, html, Input, Output, State, dash_table  # Dash core components and callbacks
import plotly.express as px  # For quick plotting
import plotly.graph_objects as go  # For detailed plot customization

# --- Machine Learning and Forecasting Libraries ---
from ultralytics import YOLO  # YOLOv8 object detection model
import pvlib  # Photovoltaic system modeling
from pvlib.location import Location  # For setting location in PVlib

# --- Utility Function: Convert Image Array to Base64 for HTML display ---
def img_arr_to_base64(arr):
    img = Image.fromarray(arr)  # Convert NumPy array to PIL Image
    buff = BytesIO()  # Create an in-memory buffer
    img.save(buff, format='PNG')  # Save image to buffer in PNG format
    b64 = base64.b64encode(buff.getvalue()).decode('ascii')  # Encode to base64 string
    return 'data:image/png;base64,' + b64  # Prefix to display in HTML

# --- Run YOLOv9 Detection for All Cities and Save Results ---
def run_inference():
    weights = "best.pt"  # Path to trained YOLO model
    model = YOLO(weights)  # Load YOLO model
    for city in ["Faro", "Porto", "Lisbon"]:
        model.predict(source=city, save=True, project="runs/detect", name=city)  # Run inference and save
        time.sleep(1)  # Small delay to ensure file writes complete

# --- Copy Original and Detected Images into Assets Folder for App Use ---
def copy_results_to_assets():
    for city in ["Faro", "Porto", "Lisbon"]:
        o_src = city  # Folder with original images
        o_dst = os.path.join("assets", city, "orig")  # Destination for original images
        if os.path.exists(o_src):
            os.makedirs(o_dst, exist_ok=True)
            for f in os.listdir(o_src):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(os.path.join(o_src, f), os.path.join(o_dst, f))

        d_src = os.path.join("runs", "detect", city)  # Folder with detected images
        d_dst = os.path.join("assets", city, "det")  # Destination for detected images
        if os.path.exists(d_src):
            os.makedirs(d_dst, exist_ok=True)
            for f in os.listdir(d_src):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    shutil.copy2(os.path.join(d_src, f), os.path.join(d_dst, f))

# --- Estimate PV Generation Curve for a Given Day and Location ---
def estimate_generation_curve(pv_count, date_str, lat=38.72, lon=-9.14, tilt=30, azimuth=180):
    module_power = 400  # Power per module in watts
    modules_per_pv = 4  # Number of modules per PV system
    efficiency = 0.18  # Efficiency of conversion
    module_area = 1.6  # Area per module in m²

    total_modules = pv_count * modules_per_pv  # Total module count
    total_area = total_modules * module_area  # Total area in m²

    date = pd.to_datetime(date_str)
    date_end = date + timedelta(days=1)

    location = Location(lat, lon, tz='UTC')  # Define PVlib location object
    times = pd.date_range(start=date, end=date_end, freq='1H', tz='UTC')[:-1]  # Hourly timestamps

    # Get clear-sky irradiance using simplified Solis model
    clearsky = location.get_clearsky(times, model='simplified_solis')
    ghi = clearsky['ghi']  # Global horizontal irradiance
    dhi = clearsky['dhi']  # Diffuse horizontal irradiance
    dni = clearsky['dni']  # Direct normal irradiance

    # Try to use real irradiance data from Open-Meteo
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={date.date()}&end_date={date.date()}"
            f"&hourly=shortwave_radiation"
            f"&timezone=auto"
        )
        res = requests.get(url, timeout=8)
        res.raise_for_status()
        data = res.json()

        irr = data['hourly']['shortwave_radiation']
        time_list = pd.to_datetime(data['hourly']['time']).tz_localize('UTC')
        df_weather = pd.DataFrame({'time': time_list, 'ghi': irr}).set_index('time')

        if not df_weather.empty:
            ghi = df_weather['ghi']
            solar_position = location.get_solarposition(df_weather.index)
            erbs = pvlib.irradiance.erbs(ghi, solar_position['apparent_zenith'])  # Split GHI into DNI and DHI
            dni = erbs['dni']
            dhi = erbs['dhi']
            times = df_weather.index
    except Exception as e:
        print("[WARNING] Failed to get real weather. Falling back to clear sky.", e)

    # Get solar position angles
    solar_position = location.get_solarposition(times)

    # Compute total irradiance on the tilted PV panel
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )

    poa_global = poa['poa_global'].fillna(0)  # Total irradiance on plane of array
    poa_global = poa_global.where(solar_position['apparent_zenith'] <= 90, 0)  # Zero if sun is below horizon

    # Estimate hourly energy generation (kWh)
    hourly_energy = poa_global * total_area * efficiency / 1000

    df_result = pd.DataFrame({
        'Hour': times.hour,
        'PV Power (kWh)': hourly_energy.round(2)  # Round for better readability
    })

    return df_result

# Create Dash app and apply Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = 'Portugal Solar Panel Detection'
app.config.suppress_callback_exceptions = True  # Allow callbacks for dynamically loaded components

# Global layout with navigation and dynamic content
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # Tracks URL for page routing
    dcc.Store(id='selected-city', data='Lisbon'),  # Store current selected city
    dcc.Store(id='img-index', data=0),  # Store current image index
    dcc.Store(id='detected-count', data=0),  # Store detected PV count

    # --- Navigation Bar with Background Image ---
    html.Div([
        # Translucent overlay to darken background image
        html.Div(style={
            'position': 'absolute', 'top': 0, 'left': 0, 'right': 0, 'bottom': 0,
            'backgroundColor': 'rgba(0, 0, 0, 0.4)', 'zIndex': 1
        }),

        # Navbar content with title and links
        dbc.NavbarSimple(
            brand=html.Div([
                # Title and authors
                html.Div([
                    html.Span("Portugal Solar Panel Detection Dashboard", style={
                        'fontSize': '1.75rem', 'fontWeight': 'bold'}),
                    html.Span("  Zhi Liu, Alex Stephan Erdfarb, Yushan Li", style={
                        'fontSize': '1rem', 'fontStyle': 'italic', 'marginLeft': '1rem'})
                ], style={'display': 'flex', 'alignItems': 'baseline'}),

                # Description text
                html.Div(
                    "We built a PV detection and forecasting system by training YOLOv9 on images from Perth, Barcelona, and Seville. "
                    "Detection was applied to Porto, Lisbon, and Faro, and generation was predicted using PVlib and Open-Meteo weather data.",
                    style={
                        'fontSize': '0.875rem', 'marginTop': '2px',
                        'color': 'white', 'maxWidth': '95ch',
                        'whiteSpace': 'normal', 'wordBreak': 'break-word'
                    }
                )
            ], style={'display': 'flex', 'flexDirection': 'column'}),

            # Navigation buttons to different pages
            children=html.Div([
                dbc.NavItem(dcc.Link("Detection", href="/", className="nav-link", style={
                    'fontSize': '1.1rem', 'padding': '6px 16px', 'color': 'white',
                    'border': '1px solid rgba(255,255,255,0.6)', 'borderRadius': '6px',
                    'backgroundColor': 'rgba(0,0,0,0.2)', 'textDecoration': 'none'
                })),
                dbc.NavItem(dcc.Link("Forecast", href="/forecast", className="nav-link", style={
                    'fontSize': '1.1rem', 'padding': '6px 16px', 'color': 'white',
                    'border': '1px solid rgba(255,255,255,0.6)', 'borderRadius': '6px',
                    'backgroundColor': 'rgba(0,0,0,0.2)', 'textDecoration': 'none'
                })),
            ], style={'position': 'absolute', 'top': '30px', 'right': '0rem', 'display': 'flex',
                      'flexDirection': 'column', 'zIndex': 3, 'gap': '0.5rem'}),

            dark=True, color=None, style={'position': 'relative', 'zIndex': 2, 'backgroundColor': 'transparent'}
        )
    ], style={
        'position': 'relative',
        'backgroundImage': 'url("/assets/Portugal.png")',
        'backgroundRepeat': 'no-repeat', 'backgroundPosition': 'center',
        'backgroundSize': '100% 100%', 'height': '140px', 'width': '100%',
        'overflow': 'hidden', 'padding': '1rem'
    }),

    html.Div(id='page-content')  # Dynamic content will be rendered here based on the URL
])

# Layout for the object detection interface
detection_layout = dbc.Container([
    dbc.Card([
        dbc.Row([
            # Map on the left side
            dbc.Col(dcc.Graph(
                id='portugal-map',
                config={'scrollZoom': True},
                style={'cursor': 'grab'}
            ), width=4),

            # Image display and control panel on the right
            dbc.Col([
                # Show original and detected images side-by-side
                dbc.Row([
                    dbc.Col(html.Div(id='original-image'), width=6),
                    dbc.Col(html.Div(id='detected-image'), width=6),
                ]),
                html.Div([
                    # Dropdowns and sliders
                    dbc.Row([
                        dbc.Col(dcc.Dropdown(
                            id='city-dropdown',
                            options=[{'label': c, 'value': c} for c in ["Lisbon", "Porto", "Faro"]],
                            value='Lisbon', clearable=False
                        ), width=4),
                        dbc.Col(html.Div([
                            dcc.Slider(id='conf-slider', min=0, max=1, step=0.1, value=0.25),
                            html.Label("Confidence", className="d-block text-center mt-1")
                        ]), width=4),
                        dbc.Col(html.Div([
                            dcc.Slider(id='iou-slider', min=0, max=1, step=0.1, value=0.45),
                            html.Label("IoU", className="d-block text-center mt-1")
                        ]), width=4),
                    ], align='center', className='mb-3'),

                    # Action buttons
                    dbc.Row([
                        dbc.Col(dbc.Button("Run Detection", id="run-btn", n_clicks=0, color="primary"), width="auto"),
                        dbc.Col(dbc.ButtonGroup([
                            dbc.Button("◀ Prev", id="prev-btn", n_clicks=0, color="secondary"),
                            dbc.Button("Next ▶", id="next-btn", n_clicks=0, color="secondary"),
                        ]), width="auto"),
                        dbc.Col(html.Div(id='detection-count', className="ms-4"), width="auto"),
                        dbc.Col(html.Div(id='latency-display', className="ms-4"), width="auto"),
                    ], align='center'),
                ], style={'marginTop': '0.8cm', 'paddingLeft': '1rem', 'paddingRight': '1rem'}),
            ], width=8),
        ], className="p-0"),
    ],
    body=True,
    className="mb-4 shadow-sm rounded-3",
    style={'marginTop': '3mm', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
], fluid=True)

# Layout for the PV generation forecast view
forecast_layout = dbc.Container([
    dbc.Card([
        # Date and slider inputs
        dbc.Row([
            dbc.Col([
                html.Label("Forecast Date"),
                dcc.DatePickerSingle(id='date-picker', date=datetime.today().date())
            ], width=2),
            dbc.Col([
                html.Label("Tilt (degrees)"),
                dcc.Slider(id='tilt-slider', min=0, max=90, step=1, value=30,
                           marks={0: '0°', 30: '30°', 60: '60°', 90: '90°'})
            ], width=5),
            dbc.Col([
                html.Label("Azimuth (degrees)"),
                dcc.Slider(id='azimuth-slider', min=0, max=360, step=10, value=180,
                           marks={0: 'N', 90: 'E', 180: 'S', 270: 'W', 360: 'N'})
            ], width=5)
        ], className="pb-4 px-3"),

        # Power table and curve graph
        dbc.Row([
            dbc.Col(
                dash_table.DataTable(
                    id='power-table',
                    style_table={'overflowY': 'auto', 'height': '400px'},
                    style_cell={'textAlign': 'center', 'fontSize': '16px', 'padding': '5px'},
                    style_header={
                        'backgroundColor': '#007AFF',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'fontSize': '14px'
                    },
                    export_format='csv'  # Allow CSV export
                ), width=3
            ),
            dbc.Col(dcc.Graph(id='pv-curve'), width=9)
        ])
    ],
    body=True,
    className="shadow-sm rounded-3",
    style={'marginTop': '3mm', 'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'})
], fluid=True)

# Callback to dynamically render the correct page layout based on URL
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/forecast':  # If URL ends with "/forecast", load forecast page
        return forecast_layout
    return detection_layout  # Default to detection page

# Callback to update selected city based on map click or dropdown selection
@app.callback(
    Output('selected-city', 'data'),  # Save selected city
    Output('city-dropdown', 'value'),  # Reflect change in dropdown
    Input('portugal-map', 'clickData'),  # Triggered by map click
    Input('city-dropdown', 'value'),  # Triggered by dropdown
    prevent_initial_call=True
)
def update_city(clickData, dropdown_value):
    ctx = dash.callback_context  # Context of what triggered the callback
    if not ctx.triggered:
        return dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]  # Who triggered?

    if trigger == 'portugal-map' and clickData:
        city_clicked = clickData['points'][0]['hovertext']  # Get city from map
        return city_clicked, city_clicked
    elif trigger == 'city-dropdown':
        return dropdown_value, dropdown_value

    return dash.no_update


@app.callback(
    Output('original-image', 'children'),  # Display original image
    Output('detected-image', 'children'),  # Display detected image with bounding boxes
    Output('img-index', 'data'),  # Update image index
    Output('detection-count', 'children'),  # Display how many PV systems detected
    Output('latency-display', 'children'),  # Show time taken to detect
    Output('detected-count', 'data'),  # Store count for forecast
    Output('portugal-map', 'figure'),  # Update city map
    Input('prev-btn', 'n_clicks'),  # Prev image button
    Input('next-btn', 'n_clicks'),  # Next image button
    Input('conf-slider', 'value'),  # YOLO confidence threshold
    Input('iou-slider', 'value'),  # YOLO IoU threshold
    Input('run-btn', 'n_clicks'),  # Trigger detection manually
    State('img-index', 'data'),  # Current image index
    State('selected-city', 'data')  # Current city
)
def update_detection(prev, nxt, conf, iou, run_clicks, idx, city):
    city = city or 'Lisbon'
    orig_dir = os.path.join('assets', city, 'orig')  # Directory for original images
    origs = sorted(os.listdir(orig_dir)) if os.path.exists(orig_dir) else []

    if not origs:
        empty = html.P('No images.')
        return empty, empty, 0, "Detection Count: 0", "Latency: 0.00s", 0, go.Figure()

    # Update image index based on button click
    idx = ((idx - 1) if dash.callback_context.triggered_id == 'prev-btn' else
           (idx + 1) if dash.callback_context.triggered_id == 'next-btn' else idx) % len(origs)

    orig_file = origs[idx]  # Current image file
    orig_path = os.path.join(orig_dir, orig_file)

    # Run YOLO detection
    start = time.time()
    model = YOLO("best.pt")
    results = model.predict(source=orig_path, conf=conf, iou=iou)
    latency = time.time() - start

    plot_arr = results[0].plot()  # Draw bounding boxes
    det_b64 = img_arr_to_base64(plot_arr)  # Convert detected image to base64
    count = len(results[0].boxes)  # Count detected objects

    total_pages = len(origs)
    page_label = f"{idx + 1}/{total_pages}"

    # Construct original image display with page indicator
    original = html.Div([
        html.H5("Original", className="text-center"),
        html.Div([
            html.Img(src=f'/assets/{city}/orig/{orig_file}', style={'width': '100%'})
        ], style={'position': 'relative'}),
        html.Div(
            page_label,
            style={
                'position': 'absolute', 'bottom': '10px', 'right': '10px',
                'color': 'white', 'backgroundColor': 'rgba(0,0,0,0.5)',
                'padding': '2px 6px', 'borderRadius': '4px', 'fontSize': '0.8rem'
            }
        )
    ], style={'position': 'relative'})

    # Construct detected image display
    detected = html.Div([
        html.H5("Detected", className="text-center"),
        html.Div(html.Img(src=det_b64, style={'width': '100%'}))
    ])

    # Build map with city markers
    df = pd.DataFrame({"City": ["Lisbon", "Porto", "Faro"],
                       "lat": [38.7223, 41.1579, 37.0194],
                       "lon": [-9.1393, -8.6291, -7.9304]})
    fig_map = px.scatter_mapbox(
        df, lat='lat', lon='lon', hover_name='City', zoom=6,
        center={'lat': 39.5, 'lon': -8},
        color_discrete_sequence=['#007AFF']
    )
    fig_map.update_layout(
        mapbox_style='open-street-map',
        margin=dict(l=0, r=0, t=0, b=0),
        height=660, width=450,
        dragmode='pan'
    )

    return original, detected, idx, f"Detection Count: {count}", f"Latency: {latency:.2f}s", count, fig_map
# Callback to compute generation forecast when PV count, date or tilt/azimuth changes
@app.callback(
    Output('pv-curve', 'figure'),  # Output plot
    Output('power-table', 'data'),  # Output table data
    Output('power-table', 'columns'),  # Output table columns
    Input('detected-count', 'data'),  # PV system count
    Input('selected-city', 'data'),  # Selected city
    Input('date-picker', 'date'),  # Forecast date
    Input('tilt-slider', 'value'),  # Panel tilt
    Input('azimuth-slider', 'value')  # Panel azimuth
)
def update_forecast(count, city, date_str, tilt, azimuth):
    city = city or 'Lisbon'
    latlon_map = {'Lisbon': (38.72, -9.14), 'Porto': (41.15, -8.62), 'Faro': (37.02, -7.93)}
    lat, lon = latlon_map.get(city, (38.72, -9.14))  # Default to Lisbon
    df_gen = estimate_generation_curve(count, date_str, lat, lon, tilt, azimuth)

    # Add total row at the bottom
    sum_row = pd.DataFrame({
        'Hour': ['Total'],
        'PV Power (kWh)': [round(df_gen['PV Power (kWh)'].sum(), 2)]
    })
    df_gen = pd.concat([df_gen, sum_row], ignore_index=True)

    # Create power curve chart (exclude total row)
    fig_curve = px.line(
        df_gen[df_gen['Hour'] != 'Total'],
        x='Hour', y='PV Power (kWh)',
        template='plotly_white',
        line_shape='spline',
        markers=True
    )
    fig_curve.update_layout(
        title=f'{city}: Estimated PV Energy Curve',
        font={'family': 'Arial', 'size': 14},
        title_font={'size': 16, 'family': 'Arial'}
    )
    fig_curve.update_xaxes(tickmode='linear', dtick=1)

    # Convert to Dash table format
    table_data = df_gen.to_dict('records')
    table_columns = [{'name': col, 'id': col} for col in df_gen.columns]

    return fig_curve, table_data, table_columns

# Entry point: runs YOLO detection once and then launches app
if __name__ == '__main__':
    run_inference()  # Run inference on all cities
    copy_results_to_assets()  # Copy detection results to assets folder
    app.run(debug=True)  # Start Dash app in debug mode
    server = app.server

