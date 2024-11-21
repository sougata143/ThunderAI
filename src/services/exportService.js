import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';

export class ExportService {
  static async exportToCSV(data, filename) {
    const csvContent = this.convertToCSV(data);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, `${filename}.csv`);
  }

  static async exportToExcel(data, filename) {
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
    XLSX.writeFile(wb, `${filename}.xlsx`);
  }

  static async exportToJSON(data, filename) {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, `${filename}.json`);
  }

  static async exportToPNG(chartRef, filename) {
    if (chartRef.current) {
      const canvas = await html2canvas(chartRef.current);
      canvas.toBlob((blob) => {
        saveAs(blob, `${filename}.png`);
      });
    }
  }

  private static convertToCSV(data) {
    const headers = Object.keys(data[0]);
    const csvRows = [
      headers.join(','),
      ...data.map(row => 
        headers.map(header => 
          JSON.stringify(row[header])
        ).join(',')
      )
    ];
    return csvRows.join('\n');
  }
} 