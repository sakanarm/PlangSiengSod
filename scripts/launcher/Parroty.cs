// Parroty.exe - a tiny launcher: starts .venv\Scripts\pythonw.exe -m rtvc from the folder this exe is in.
// Built locally by scripts\build_launcher.ps1 with the C# compiler that ships with Windows (.NET Framework 4).
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Windows.Forms;

[assembly: AssemblyTitle("Parroty")]
[assembly: AssemblyProduct("Parroty - real-time voice changer")]
[assembly: AssemblyDescription("Starts Parroty from its project folder")]
[assembly: AssemblyVersion("0.2.0.0")]

static class Program
{
    [STAThread]
    static int Main()
    {
        string root = AppDomain.CurrentDomain.BaseDirectory;
        string pyw = Path.Combine(root, @".venv\Scripts\pythonw.exe");
        if (!File.Exists(pyw))
        {
            MessageBox.Show("Parroty is not set up yet.\n\nRun this once in PowerShell, in the Parroty folder:\n\n"
                            + "    powershell -ExecutionPolicy Bypass -File .\\setup.ps1\n\nDetails: docs\\INSTALL_WINDOWS.md",
                            "Parroty", MessageBoxButtons.OK, MessageBoxIcon.Warning);
            return 1;
        }
        try
        {
            Process.Start(new ProcessStartInfo(pyw, "-m rtvc") { WorkingDirectory = root, UseShellExecute = false });
            return 0;
        }
        catch (Exception e)
        {
            MessageBox.Show("Could not start Parroty:\n\n" + e.Message, "Parroty", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
