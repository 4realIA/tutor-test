//! Facts about the computer, for the report and the disk-space check.

use std::path::{Path, PathBuf};
use std::process::Command;

use sysinfo::{Disks, System};

use crate::results::Machine;

pub fn machine() -> Machine {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|b| !b.is_empty())
        .unwrap_or_else(|| "Unknown".into());
    Machine {
        os: System::long_os_version().unwrap_or_else(|| std::env::consts::OS.into()),
        cpu,
        cores: sys.cpus().len(),
        ram_gb: sys.total_memory() as f64 / 1024f64.powi(3),
        gpus: gpus(),
    }
}

/// Graphics adapters as Windows names them; on Macs the GPU is part of the chip.
fn gpus() -> Vec<String> {
    if cfg!(target_os = "windows") {
        let out = Command::new("powershell")
            .args([
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ])
            .output();
        match out {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect(),
            _ => Vec::new(),
        }
    } else if cfg!(target_os = "macos") {
        vec!["Built into the Apple chip".into()]
    } else {
        Vec::new()
    }
}

/// Free space on the disk holding `path`, in bytes.
pub fn free_space(path: &Path) -> Option<u64> {
    let path = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let disks = Disks::new_with_refreshed_list();
    disks
        .list()
        .iter()
        .filter(|d| path.starts_with(d.mount_point()))
        .max_by_key(|d| d.mount_point().as_os_str().len())
        .map(|d| d.available_space())
}

/// The user's home folder.
pub fn home() -> PathBuf {
    std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
}

/// The folder holding this program.
pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(Path::to_path_buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn machine_has_basics() {
        let m = machine();
        assert!(m.cores > 0);
        assert!(m.ram_gb > 0.0);
        assert!(!m.os.is_empty());
    }

    #[test]
    fn free_space_of_temp_dir() {
        assert!(free_space(&std::env::temp_dir()).is_some_and(|b| b > 0));
    }
}
