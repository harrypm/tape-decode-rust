use std::fs::File;
use std::io;
use std::io::{stderr, stdin, stdout};

#[cfg(unix)]
mod unix {
    use std::fs::File;
    use std::io;
    use std::os::fd::AsRawFd;
    use std::os::fd::{FromRawFd, RawFd};

    pub(super) fn dup_as_file<T>(file_like: T) -> io::Result<File>
    where
        T: AsRawFd,
    {
        let fd = file_like.as_raw_fd();
        let new_fd: RawFd = unsafe { libc::dup(fd) };
        if new_fd < 0 {
            return Err(io::Error::last_os_error());
        }

        Ok(unsafe { File::from_raw_fd(new_fd) })
    }
}

#[cfg(unix)]
use unix as imp;

#[cfg(windows)]
mod windows {
    use std::fs::File;
    use std::io;
    use std::os::windows::io::{FromRawHandle, RawHandle};
    use std::os::windows::prelude::AsRawHandle;

    pub(super) fn dup_as_file<T>(file_like: T) -> io::Result<File>
    where
        T: AsRawHandle,
    {
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::Foundation::{DuplicateHandle, DUPLICATE_SAME_ACCESS};

        let handle = file_like.as_raw_handle();
        if handle.is_null() {
            return Err(io::Error::other("standard handle is null"));
        }

        let current_process: HANDLE = -1isize as HANDLE;
        let mut duplicated: HANDLE = std::ptr::null_mut();

        let ok = unsafe {
            DuplicateHandle(
                current_process,
                handle as HANDLE,
                current_process,
                &mut duplicated,
                0,
                0,
                DUPLICATE_SAME_ACCESS,
            )
        };

        if ok == 0 {
            return Err(io::Error::last_os_error());
        }

        if duplicated.is_null() {
            return Err(io::Error::other("duplicated standard handle is null"));
        }

        Ok(unsafe { File::from_raw_handle(duplicated as RawHandle) })
    }
}

#[cfg(windows)]
use windows as imp;

pub fn stdin_file() -> io::Result<File> {
    imp::dup_as_file(stdin())
}

pub fn stdout_file() -> io::Result<File> {
    imp::dup_as_file(stdout())
}

#[allow(dead_code)]
pub fn stderr_file() -> io::Result<File> {
    imp::dup_as_file(stderr())
}
