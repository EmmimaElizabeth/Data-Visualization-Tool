from flask import Flask, render_template, request, send_file, jsonify, make_response
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import io
import re
import os
import threading
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

thread_local = threading.local()

class DataProcessor:
    def __init__(self):
        self.df = None
        self.date_column = None
        self.min_date = None
        self.max_date = None
        self.y_axis_columns = []
        self.file_loaded = False

    def handle_file_select(self, file_path):
        try:
            # Read the file
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            # Try to identify and convert the date column
            for column in self.df.columns:
                try:
                    self.df[column] = pd.to_datetime(self.df[column], format='mixed')
                    self.date_column = column
                    break
                except (ValueError, TypeError):
                    continue
            
            if self.date_column is None:
                raise ValueError("No valid date/time column found in the file")
            
            self.df = self.df.sort_values(by=self.date_column)
            self.min_date = self.df[self.date_column].min().date()
            self.max_date = self.df[self.date_column].max().date()
            
            # Filter out the date column from y-axis options
            self.y_axis_columns = [col for col in self.df.columns if col != self.date_column]
            self.file_loaded = True
            
        except Exception as e:
            self.file_loaded = False
            raise ValueError(f"Error processing file: {str(e)}")

    def get_plot_data(self):
        if not self.file_loaded:
            raise ValueError("No file has been loaded or file processing failed")
        
        date_range = {
            'min': self.min_date.strftime('%Y-%m-%d'),
            'max': self.max_date.strftime('%Y-%m-%d')
        }
        return self.y_axis_columns, date_range

    def plot_data(self, start_date, end_date, y_axis):
        if not self.file_loaded or self.df is None or self.date_column is None:
            raise ValueError("No data available to plot")

        try:
            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            if start_date < self.min_date or end_date > self.max_date:
                raise ValueError(f"Selected dates must be between {self.min_date} and {self.max_date}")

            mask = (self.df[self.date_column].dt.date >= start_date) & (self.df[self.date_column].dt.date <= end_date)
            filtered_df = self.df.loc[mask]

            filtered_df[y_axis] = pd.to_numeric(filtered_df[y_axis], errors='coerce')
            
            y_max = filtered_df[y_axis].max()
            y_min = filtered_df[y_axis].min()

            plt.figure(figsize=(12, 6))
            plt.plot(filtered_df[self.date_column], filtered_df[y_axis], linestyle='-', linewidth=1, color='red')
            plt.title(f'Time Series Plot of {y_axis} ({start_date} to {end_date})')
            plt.xlabel('Date')
            plt.ylabel(y_axis)

            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())

            plt.xticks(rotation=45, ha='right')
            
            y_range = y_max - y_min
            y_min_padded = y_min - 0.1*y_range
            y_max_padded = y_max + 0.1*y_range
            plt.ylim([y_min_padded, y_max_padded])
            
            plt.subplots_adjust(bottom=0.25)

            plt.figtext(0.95, 0.01, f'Min: {y_min:.2f} | Max: {y_max:.2f}', 
                ha='right', fontsize=10, color='blue')

            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=300)
            img.seek(0)
            plt.close()
            return img
        except Exception as e:
            plt.close()
            raise ValueError(f"Error generating plot: {str(e)}")

class ContourPlotter:
    def __init__(self):
        self.df = None
        self.date_column = None
        self.min_date = None
        self.max_date = None
        self.variable_groups = {}
        self.file_loaded = False

    def process_file(self, file_path):
        try:
            if file_path.endswith('.csv'):
                self.df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.df = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format")
            
            if 'Time' not in self.df.columns:
                raise ValueError("Required 'Time' column not found in the file")

            self.df['Time'] = pd.to_datetime(self.df['Time'], format='mixed')
            self.date_column = 'Time'
            columns = self.df.columns
            self.variable_groups = self.group_similar_columns(columns)
            self.min_date = self.df['Time'].min().date()
            self.max_date = self.df['Time'].max().date()
            self.file_loaded = True
            
        except Exception as e:
            self.file_loaded = False
            raise ValueError(f"Error processing file: {str(e)}")

    def extract_depth(self, col):
        match = re.search(r'(\d+\.?\d*)', col)
        return float(match.group(1)) if match else None

    def extract_main_variable(self, col):
        main_variable = re.sub(r'\d+', '', col)
        main_variable = re.sub(r'[^a-zA-Z]', '', main_variable)
        return main_variable.lower()

    def are_variables_similar(self, var1, var2):
        if var1 == var2:
            return True
        if var1.replace('dvs', '') == var2 or var2.replace('dvs', '') == var1:
            return True
        return False

    def group_similar_columns(self, columns):
        variable_groups = {}
        for col in columns:
            if col == 'Time':
                continue
            main_variable = self.extract_main_variable(col)
            matched_group = None
            for existing_variable in variable_groups:
                if self.are_variables_similar(main_variable, existing_variable):
                    matched_group = existing_variable
                    break
            if matched_group:
                variable_groups[matched_group].append(col)
            else:
                variable_groups[main_variable] = [col]
        return {k: v for k, v in variable_groups.items() if len(v) > 1}

    def get_variables(self):
        if not self.file_loaded:
            raise ValueError("No file has been loaded")
        return list(self.variable_groups.keys())

    def get_date_range(self):
        if not self.file_loaded:
            raise ValueError("No file has been loaded")
        return {
            'min': self.min_date.strftime('%Y-%m-%d'),
            'max': self.max_date.strftime('%Y-%m-%d')
        }

    def plot_variable(self, variable, start_date, end_date):
        if not self.file_loaded or variable not in self.variable_groups:
            raise ValueError("Invalid variable or no data available")

        try:
            columns = self.variable_groups[variable]
            depths = [self.extract_depth(col) for col in columns]
            
            sorted_indices = np.argsort(depths)[::-1]
            sorted_depths = np.array(depths)[sorted_indices]
            sorted_columns = np.array(columns)[sorted_indices]

            start_date = pd.to_datetime(start_date).date()
            end_date = pd.to_datetime(end_date).date()

            data = self.df[(self.df['Time'].dt.date >= start_date) & (self.df['Time'].dt.date <= end_date)]
            data[sorted_columns] = data[sorted_columns].fillna(np.nan)
            
            time = data['Time']
            time_numeric = mdates.date2num(time)
            variable_data = data[sorted_columns].values.T
            
            mask = ~np.isnan(variable_data).all(axis=0)
            time_filtered = time_numeric[mask]
            variable_filtered = variable_data[:, mask]

            depth_indices = np.arange(len(sorted_depths))
            X, Y = np.meshgrid(time_filtered, depth_indices)

            plt.figure(figsize=(14, 8))
            
            contour = plt.contourf(X, Y, variable_filtered, levels=np.linspace(np.nanmin(variable_filtered), np.nanmax(variable_filtered), 100), cmap='jet')
            
            plt.gca().set_yticks(depth_indices)
            plt.gca().set_yticklabels([f'{depth:.1f}' for depth in sorted_depths])
            plt.gca().invert_yaxis()
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            
            plt.colorbar(contour, label=variable)
            
            plt.title(f'Contour Plot of {variable} ({start_date} to {end_date})')
            plt.xlabel('Date')
            plt.ylabel('Depth')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            img = io.BytesIO()
            plt.savefig(img, format='png', dpi=300)
            img.seek(0)
            plt.close()
            return img
        except Exception as e:
            plt.close()
            raise ValueError(f"Error generating contour plot: {str(e)}")

def get_data_processor():
        if not hasattr(thread_local, "data_processor"):
            thread_local.data_processor = DataProcessor()
        return thread_local.data_processor

def get_contour_plotter():
        if not hasattr(thread_local, "contour_plotter"):
            thread_local.contour_plotter = ContourPlotter()
        return thread_local.contour_plotter

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('main.html', error="No file part")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('main.html', error="No file selected")
        
        try:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            
            data_processor = get_data_processor()
            contour_plotter = get_contour_plotter()
            
            data_processor.handle_file_select(file_path)
            contour_plotter.process_file(file_path)
            
            return render_template('select.html', filename=file.filename)
            
        except Exception as e:
            return render_template('main.html', error=str(e))
    
    return render_template('main.html')

@app.route('/select_plot', methods=['GET', 'POST'])
def select_plot():
    if request.method == 'POST':
        try:
            plot_type = request.form.get('plot_type')
            filename = request.form.get('filename')
            
            data_processor = get_data_processor()
            contour_plotter = get_contour_plotter()
            
            if plot_type == 'plot_time_series':
                if not data_processor.file_loaded:
                    raise ValueError("No data loaded for time series plot")
                columns, date_range = data_processor.get_plot_data()
                return render_template('time.html', columns=columns, filename=filename, date_range=date_range)
            elif plot_type == 'plot_contour':
                if not contour_plotter.file_loaded:
                    raise ValueError("No data loaded for contour plot")
                variables = contour_plotter.get_variables()
                date_range = contour_plotter.get_date_range()
                return render_template('contour.html', variables=variables, filename=filename, date_range=date_range)
            else:
                raise ValueError("Invalid plot type selected")
                
        except Exception as e:
            return render_template('select.html', error=str(e))
    
    return render_template('select.html')

@app.route('/plot_time_series', methods=['GET','POST'])
def plot_time_series():
    try:
        if request.method == 'POST':
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            y_axis = request.form.get('y_axis')
            
            print(f"Start Date: {start_date}, End Date: {end_date}, Y Axis: {y_axis}")
        
        if not all([start_date, end_date, y_axis]):
            raise ValueError("Missing required parameters")
        
        data_processor = get_data_processor()
        plot_data = data_processor.plot_data(start_date, end_date, y_axis)
        return send_file(plot_data, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
    return render_template('time.html')

@app.route('/plot_contour', methods=['GET','POST'])
def plot_contour():
    try:
        if request.method == 'POST':
            start_date = request.form.get('start_date')
            end_date = request.form.get('end_date')
            variable = request.form.get('variable')
        
        if not all([start_date, end_date, variable]):
            raise ValueError("Missing required parameters")
        
        contour_plotter = get_contour_plotter()
        plot_data = contour_plotter.plot_variable(variable, start_date, end_date)
        return send_file(plot_data, mimetype='image/png')
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    return render_template('contour.html')

@app.route('/download_plot', methods=['GET'])
def download_plot():
    try:
        plot_type = request.args.get('plot_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        
        if plot_type == 'time_series':
            y_axis = request.args.get('y_axis')
            if not all([start_date, end_date, y_axis]):
                raise ValueError("Missing required parameters for time series plot")
            
            data_processor = get_data_processor()
            plot_data = data_processor.plot_data(start_date, end_date, y_axis)
            
            response = make_response(plot_data.getvalue())
            response.headers['Content-Type'] = 'image/png'
            response.headers['Content-Disposition'] = f'attachment; filename=time_series_{y_axis}_{start_date}_{end_date}.png'
            return response
            
        elif plot_type == 'contour':
            variable = request.args.get('variable')
            if not all([start_date, end_date, variable]):
                raise ValueError("Missing required parameters for contour plot")
            
            contour_plotter = get_contour_plotter()
            plot_data = contour_plotter.plot_variable(variable, start_date, end_date)
            
            response = make_response(plot_data.getvalue())
            response.headers['Content-Type'] = 'image/png'
            response.headers['Content-Disposition'] = f'attachment; filename=contour_{variable}_{start_date}_{end_date}.png'
            return response 
            
        else:
            raise ValueError("Invalid plot type")
            
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/clear_data', methods=['POST'])
def clear_data():
    try:
        # Clear the thread-local storage
        if hasattr(thread_local, "data_processor"):
            delattr(thread_local, "data_processor")
        if hasattr(thread_local, "contour_plotter"):
            delattr(thread_local, "contour_plotter")
            
        # Clean up files in upload folder
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                
        return jsonify({'message': 'Data cleared successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.errorhandler(404) 
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    app.run(debug=True)
    
