import { saveAs } from 'file-saver';
import * as XLSX from 'xlsx';

interface ExportOptions {
  filename: string;
  format: 'csv' | 'excel' | 'json';
}

export class ExportService {
  static async exportToCSV(data: any[], filename: string): Promise<void> {
    const csvContent = this.convertToCSV(data);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    saveAs(blob, `${filename}.csv`);
  }

  static async exportToExcel(data: any[], filename: string): Promise<void> {
    const ws = XLSX.utils.json_to_sheet(data);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Sheet1');
    XLSX.writeFile(wb, `${filename}.xlsx`);
  }

  static async exportToJSON(data: any[], filename: string): Promise<void> {
    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    saveAs(blob, `${filename}.json`);
  }

  private static convertToCSV(data: any[]): string {
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