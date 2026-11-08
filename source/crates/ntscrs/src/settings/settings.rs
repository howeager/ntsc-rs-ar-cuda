//! This is used to dynamically inform API consumers of the settings that can be passed to ntsc-rs. This lets various
//! UIs and effect plugins to query this set of settings and display them in their preferred format without having to
//! duplicate a bunch of code.
// TODO: replace with a bunch of metaprogramming macro magic?

use core::{
    borrow::Borrow,
    error::Error,
    fmt::{Display, Write as _},
    hash::{Hash, Hasher},
    num::{ParseFloatError, ParseIntError},
    ops::RangeInclusive,
};

use alloc::{borrow::Cow, boxed::Box, string::String, vec, vec::Vec};

use hifijson::{
    Expect, SliceLexer,
    num::LexWrite as _,
    str::{Lex as _, LexAlloc as _},
    token::Lex,
};
use num_enum::TryFromPrimitive;

// These are the individual setting definitions. The descriptions of what they do are included below, so I mostly won't
// repeat them here.

/// The "full settings" equivalent of an `Option<T>` for an optionally-disabled section of the settings.
#[derive(Debug, Clone, PartialEq)]
pub struct SettingsBlock<T> {
    pub enabled: bool,
    pub settings: T,
}

impl<T> SettingsBlock<T> {
    pub(crate) const fn enabled(inner: T) -> Self {
        Self {
            enabled: true,
            settings: inner,
        }
    }

    pub fn as_option(&self) -> Option<&T> {
        if self.enabled {
            Some(&self.settings)
        } else {
            None
        }
    }
}

impl<T: Default + Clone> From<&Option<T>> for SettingsBlock<T> {
    fn from(opt: &Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: match opt {
                Some(v) => v.clone(),
                None => T::default(),
            },
        }
    }
}

impl<T: Default> From<Option<T>> for SettingsBlock<T> {
    fn from(opt: Option<T>) -> Self {
        Self {
            enabled: opt.is_some(),
            settings: opt.unwrap_or_default(),
        }
    }
}

impl<T> From<SettingsBlock<T>> for Option<T> {
    fn from(value: SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings)
        } else {
            None
        }
    }
}

impl<T: Clone> From<&SettingsBlock<T>> for Option<T> {
    fn from(value: &SettingsBlock<T>) -> Self {
        if value.enabled {
            Some(value.settings.clone())
        } else {
            None
        }
    }
}

impl<T: Default> Default for SettingsBlock<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            settings: T::default(),
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EnumValue(pub u32);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnySetting {
    Enum(u32),
    Int(i32),
    Float(f32),
    Bool(bool),
}

impl AnySetting {
    pub fn type_name(&self) -> &'static str {
        match self {
            AnySetting::Enum(_) => "enum",
            AnySetting::Int(_) => "i32 or u32",
            AnySetting::Float(_) => "f32",
            AnySetting::Bool(_) => "bool",
        }
    }
}

pub trait SettingField: Sized {
    fn downcast(value: &AnySetting) -> Option<Self>;
    fn upcast(self) -> AnySetting;
}

impl<U: TryFrom<u32> + Into<u32>, T: SettingsEnum + TryFromPrimitive<Primitive = U> + Into<U>>
    SettingField for T
{
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Enum(e) => T::try_from_primitive((*e).try_into().ok()?).ok(),
            _ => None,
        }
    }

    fn upcast(self) -> AnySetting {
        AnySetting::Enum(<Self as Into<U>>::into(self).into())
    }
}

impl SettingField for i32 {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Int(i) => Some(*i),
            _ => None,
        }
    }

    fn upcast(self) -> AnySetting {
        AnySetting::Int(self)
    }
}

impl SettingField for EnumValue {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Enum(e) => Some(Self(*e)),
            _ => None,
        }
    }

    fn upcast(self) -> AnySetting {
        AnySetting::Enum(self.0)
    }
}

impl SettingField for f32 {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn upcast(self) -> AnySetting {
        AnySetting::Float(self)
    }
}

impl SettingField for bool {
    fn downcast(value: &AnySetting) -> Option<Self> {
        match value {
            AnySetting::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn upcast(self) -> AnySetting {
        AnySetting::Bool(self)
    }
}

pub trait SettingsEnum {}

/// A fixed identifier that points to a given setting. The id and name cannot be changed or reused once created.
#[derive(Clone)]
pub struct SettingID<T: Settings> {
    pub id: u32,
    pub name: &'static str,
    pub get: fn(settings: &T) -> AnySetting,
    pub set: fn(settings: &mut T, value: AnySetting) -> Result<(), GetSetFieldError>,
}

impl<T: Settings> core::fmt::Debug for SettingID<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SettingID")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("get", &self.get)
            .field("set", &self.set)
            .finish()
    }
}

// We can't use derive here because of the type parameter:
// https://github.com/rust-lang/rust/issues/26925
impl<T: Settings> Hash for SettingID<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
        self.name.hash(state);
        self.get.hash(state);
        self.set.hash(state);
    }
}

impl<T: Settings> PartialEq for SettingID<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id && self.name == other.name
    }
}

impl<T: Settings> Eq for SettingID<T> {}

impl<T: Settings> SettingID<T> {
    pub const fn new(
        id: u32,
        name: &'static str,
        get: fn(settings: &T) -> AnySetting,
        set: fn(settings: &mut T, value: AnySetting) -> Result<(), GetSetFieldError>,
    ) -> Self {
        Self { id, name, get, set }
    }
}

#[macro_export]
macro_rules! setting_id {
    ($id:expr, $name:expr, $($field_path:ident).+) => {
        $crate::settings::SettingID::new(
            $id,
            $name,
            |settings| $crate::settings::SettingField::upcast(settings.$($field_path).+),
            |settings, value| {
                settings.$($field_path).+ = $crate::settings::SettingField::downcast(&value).ok_or_else(|| $crate::settings::GetSetFieldError::TypeMismatch {
                    actual_type: value.type_name(),
                    requested_type: core::any::type_name_of_val(&settings.$($field_path).+)
                })?;
                Ok(())
            }
        )
    }
}

/// Menu item for a SettingKind::Enumeration.
#[derive(Debug, Clone)]
pub struct MenuItem {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub index: u32,
}

/// All of the types a setting can take. API consumers can map this to the UI elements available in whatever they're
/// porting ntsc-rs to.
#[derive(Debug, Clone)]
pub enum SettingKind<T: Settings> {
    /// Selection of specific options, preferably in a specific order.
    Enumeration { options: Vec<MenuItem> },
    /// Range from 0% to 100%.
    Percentage { logarithmic: bool },
    /// Inclusive discrete (integer) range.
    IntRange { range: RangeInclusive<i32> },
    /// Inclusive continuous range.
    FloatRange {
        range: RangeInclusive<f32>,
        logarithmic: bool,
    },
    /// Boolean/checkbox.
    Boolean,
    /// Group of settings, which contains an "enable/disable" checkbox and child settings.
    Group { children: Vec<SettingDescriptor<T>> },
}

#[derive(Clone, Copy, Debug)]
pub enum GetSetFieldError {
    TypeMismatch {
        actual_type: &'static str,
        requested_type: &'static str,
    },
    NoSuchID(&'static str),
}

impl Display for GetSetFieldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GetSetFieldError::TypeMismatch {
                actual_type,
                requested_type,
            } => write!(
                f,
                "Tried to get or set field with type {requested_type}, but actual type is \
                 {actual_type}"
            ),
            GetSetFieldError::NoSuchID(id) => write!(f, "No such field with ID {id}"),
        }
    }
}

pub trait Settings: Default {
    fn get_field<T: 'static + SettingField>(
        &self,
        id: &SettingID<Self>,
    ) -> Result<T, GetSetFieldError> {
        let value = (id.get)(self);
        SettingField::downcast(&value).ok_or_else(|| GetSetFieldError::TypeMismatch {
            actual_type: value.type_name(),
            requested_type: core::any::type_name::<T>(),
        })
    }

    fn set_field<T: 'static + SettingField>(
        &mut self,
        id: &SettingID<Self>,
        value: T,
    ) -> Result<(), GetSetFieldError> {
        (id.set)(self, value.upcast())
    }

    /// Returns settings which e.g. new presets can be applied on top of without any newly-added settings having an
    /// additional effect on the result. For example, a new setting added to an existing group would probably be set to
    /// 0, whereas an entirely new settings group could have all its settings at nice defaults but simply be disabled.
    /// Settings that have always existed can take on their regular default values, which are subject to change.
    fn legacy_value() -> Self;

    fn setting_descriptors() -> Box<[SettingDescriptor<Self>]>;
}

/// A single setting, which includes the data common to all settings (its name, optional description/tooltip, and ID)
/// along with a SettingKind which contains data specific to the type of setting.
#[derive(Debug, Clone)]
pub struct SettingDescriptor<T: Settings> {
    pub label: &'static str,
    pub description: Option<&'static str>,
    pub kind: SettingKind<T>,
    pub id: SettingID<T>,
}

#[derive(Debug)]
pub enum ParseSettingsError {
    InvalidJSON(hifijson::Error),
    ParseFloat(ParseFloatError),
    ParseInt(ParseIntError),
    MissingField { field: &'static str },
    WrongApplication,
    UnsupportedVersion { version: f32 },
    InvalidSettingType { key: String, expected: &'static str },
    GetSetField(GetSetFieldError),
}

impl Display for ParseSettingsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ParseSettingsError::InvalidJSON(e) => e.fmt(f),
            ParseSettingsError::ParseFloat(e) => e.fmt(f),
            ParseSettingsError::ParseInt(e) => e.fmt(f),
            ParseSettingsError::MissingField { field } => {
                write!(f, "Missing field: {}", field)
            }
            ParseSettingsError::WrongApplication => {
                write!(f, "ntscQT presets are not supported")
            }
            ParseSettingsError::UnsupportedVersion { version } => {
                write!(f, "Unsupported version: {}", version)
            }
            ParseSettingsError::InvalidSettingType { key, expected } => {
                write!(f, "Setting {} is not a(n) {}", key, expected)
            }
            ParseSettingsError::GetSetField(e) => {
                write!(f, "Error getting or setting field: {}", e)
            }
        }
    }
}

impl Error for ParseSettingsError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}

impl From<hifijson::Error> for ParseSettingsError {
    fn from(err: hifijson::Error) -> Self {
        Self::InvalidJSON(err)
    }
}

impl From<GetSetFieldError> for ParseSettingsError {
    fn from(err: GetSetFieldError) -> Self {
        Self::GetSetField(err)
    }
}

impl From<ParseFloatError> for ParseSettingsError {
    fn from(value: ParseFloatError) -> Self {
        Self::ParseFloat(value)
    }
}

impl From<ParseIntError> for ParseSettingsError {
    fn from(value: ParseIntError) -> Self {
        Self::ParseInt(value)
    }
}

pub(super) trait FromValue: Sized {
    fn from_value(value: &JsonValue<'_>) -> Result<Self, ParseSettingsError>;
}

impl FromValue for f32 {
    fn from_value(value: &JsonValue<'_>) -> Result<Self, ParseSettingsError> {
        match value {
            JsonValue::Bool(_) => Err(ParseSettingsError::GetSetField(
                GetSetFieldError::TypeMismatch {
                    actual_type: "boolean",
                    requested_type: "number",
                },
            )),
            JsonValue::Number(n) => Ok(n.parse()?),
        }
    }
}

impl FromValue for i32 {
    fn from_value(value: &JsonValue<'_>) -> Result<Self, ParseSettingsError> {
        match value {
            JsonValue::Bool(_) => Err(ParseSettingsError::GetSetField(
                GetSetFieldError::TypeMismatch {
                    actual_type: "boolean",
                    requested_type: "number",
                },
            )),
            JsonValue::Number(n) => Ok(n
                .parse::<i32>()
                .or_else(|_| n.parse::<f64>().map(|n| n as i32))?),
        }
    }
}

impl FromValue for u32 {
    fn from_value(value: &JsonValue<'_>) -> Result<Self, ParseSettingsError> {
        match value {
            JsonValue::Bool(_) => Err(ParseSettingsError::GetSetField(
                GetSetFieldError::TypeMismatch {
                    actual_type: "boolean",
                    requested_type: "number",
                },
            )),
            JsonValue::Number(n) => {
                Ok(n.parse().or_else(|_| n.parse::<f64>().map(|n| n as u32))?)
            }
        }
    }
}

impl FromValue for bool {
    fn from_value(value: &JsonValue<'_>) -> Result<Self, ParseSettingsError> {
        match value {
            JsonValue::Bool(b) => Ok(*b),
            JsonValue::Number(_) => Err(ParseSettingsError::GetSetField(
                GetSetFieldError::TypeMismatch {
                    actual_type: "number",
                    requested_type: "boolean",
                },
            )),
        }
    }
}

/// Sorted key-value map for JSON parsing. Items are indexed via binary search, and the last item wins in the event of a
/// tie.
pub(super) struct SortedMap<K: Ord, V> {
    items: Vec<(K, V)>,
}

impl<K: Ord, V> SortedMap<K, V> {
    pub fn new(mut items: Vec<(K, V)>) -> Self {
        items.sort_by(|a, b| a.0.cmp(&b.0));
        Self { items }
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        // binary_search_by returns an arbitrary result if there are multiple matches; we want to match the behavior of
        // a hash map (last inserted wins) and return the last one.
        let upper_bound = self.items.partition_point(|(k, _)| k.borrow() <= key);
        let (item_k, item_v) = self.items.get(upper_bound.checked_sub(1)?)?;
        (item_k.borrow() == key).then_some(item_v)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q> + Ord,
        Q: Ord + ?Sized,
    {
        self.items
            .binary_search_by(|a| a.0.borrow().cmp(key))
            .is_ok()
    }
}

/// Convenience trait for asserting the "shape" of the JSON we're parsing is what we expect.
pub(super) trait GetAndExpect {
    fn get_and_expect<T: FromValue + Clone>(
        &self,
        key: &str,
    ) -> Result<Option<T>, ParseSettingsError>;
}

impl<'a> GetAndExpect for SortedMap<Cow<'a, str>, JsonValue<'a>> {
    fn get_and_expect<T: FromValue + Clone>(
        &self,
        key: &str,
    ) -> Result<Option<T>, ParseSettingsError> {
        self.get(key).map(|v| T::from_value(v)).transpose()
    }
}

/// Introspectable list of settings and their types and ranges.
#[derive(Debug, Clone)]
pub struct SettingsList<T: Settings> {
    pub setting_descriptors: Box<[SettingDescriptor<T>]>,
}

#[derive(Debug)]
pub enum SerializeSettingsError {
    InvalidKeyCharacter {
        key: &'static str,
    },
    InvalidFloat {
        key: &'static str,
        value: f32,
    },
    FormatError(core::fmt::Error),
    #[cfg(feature = "std")]
    IoError(std::io::Error),
    #[cfg(not(feature = "std"))]
    IoError,
}

impl core::fmt::Display for SerializeSettingsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SerializeSettingsError::InvalidKeyCharacter { key } => {
                write!(f, "invalid character in key \"{key}\"")
            }
            SerializeSettingsError::InvalidFloat { key, value } => {
                write!(f, "invalid float value {value} for setting \"{key}\"")
            }
            SerializeSettingsError::FormatError(e) => write!(f, "error formatting value: {e}"),
            #[cfg(feature = "std")]
            SerializeSettingsError::IoError(e) => write!(f, "I/O error: {e}"),
            #[cfg(not(feature = "std"))]
            SerializeSettingsError::IoError => {
                // This can't happen but panics increase code size
                write!(f, "I/O error")
            }
        }
    }
}

impl Error for SerializeSettingsError {}

impl From<core::fmt::Error> for SerializeSettingsError {
    fn from(value: core::fmt::Error) -> Self {
        Self::FormatError(value)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for SerializeSettingsError {
    fn from(value: std::io::Error) -> Self {
        Self::IoError(value)
    }
}

pub(super) enum JsonValue<'a> {
    Bool(bool),
    Number(&'a str),
}

fn parse<'a>(
    next: u8,
    lexer: &mut SliceLexer<'a>,
) -> Result<Option<JsonValue<'a>>, hifijson::Error> {
    let nob = |o: Option<bool>| o.map(JsonValue::Bool);
    match next {
        b'a'..=b'z' => Ok(lexer.null_or_bool().map(nob).ok_or(Expect::Value)?),
        b'0'..=b'9' | b'-' => Ok(Some(JsonValue::Number(lexer.num_string().validated()?.0))),
        // We have no string settings
        b'"' => Ok({
            lexer.str_ignore().map_err(hifijson::Error::Str)?;
            None
        }),
        // We have no array settings
        b'[' => Ok({
            lexer
                .discarded()
                .seq(b']', SliceLexer::ws_peek, |next, lexer| {
                    parse(next, lexer).map(|_| ())
                })?;
            None
        }),
        // All settings are in the top level; objects are ignored
        b'{' => Ok({
            lexer
                .discarded()
                .seq(b'}', SliceLexer::ws_peek, |next, lexer| {
                    lexer.expect(|_| Some(next), b'"').ok_or(Expect::String)?;
                    lexer.str_ignore().map_err(hifijson::Error::Str)?;
                    lexer
                        .expect(SliceLexer::ws_peek, b':')
                        .ok_or(Expect::Colon)?;
                    parse(lexer.ws_peek().ok_or(Expect::Value)?, lexer)?;
                    Ok::<_, hifijson::Error>(())
                })?;
            None
        }),
        _ => Err(Expect::Value)?,
    }
}

fn parse_root_object<'a>(
    next: u8,
    lexer: &mut SliceLexer<'a>,
) -> Result<SortedMap<Cow<'a, str>, JsonValue<'a>>, hifijson::Error> {
    match next {
        b'{' => Ok({
            let mut entries = Vec::new();
            lexer
                .discarded()
                .seq(b'}', SliceLexer::ws_peek, |next, lexer| {
                    lexer.expect(|_| Some(next), b'"').ok_or(Expect::String)?;
                    let key = lexer.str_string().map_err(hifijson::Error::Str)?;
                    lexer
                        .expect(SliceLexer::ws_peek, b':')
                        .ok_or(Expect::Colon)?;
                    let value = parse(lexer.ws_peek().ok_or(Expect::Value)?, lexer)?;
                    if let Some(value) = value {
                        entries.push((key, value));
                    }
                    Ok::<_, hifijson::Error>(())
                })?;
            SortedMap::new(entries)
        }),
        _ => Err(Expect::Value)?,
    }
}

pub(super) fn parse_json<'a>(
    json: &'a str,
) -> Result<SortedMap<Cow<'a, str>, JsonValue<'a>>, ParseSettingsError> {
    let mut lexer = SliceLexer::new(json.as_bytes());
    Ok(lexer.exactly_one(Lex::ws_peek, parse_root_object)?)
}

impl<T: Settings> SettingsList<T> {
    /// Construct a list of all the effect settings. This isn't meant to be mutated--you should just create one instance
    /// of this to use for your entire application/plugin.
    pub fn new() -> Self {
        Self {
            setting_descriptors: T::setting_descriptors(),
        }
    }

    fn stream_json<F: for<'a> FnMut(&'a str) -> Result<(), SerializeSettingsError>>(
        &self,
        settings: &T,
        mut emit: F,
    ) -> Result<(), SerializeSettingsError> {
        emit("{")?;

        let mut fmt_buf = String::with_capacity(32);

        for descriptor in self.all_descriptors() {
            // key
            emit("\"")?;
            emit(descriptor.id.name)?;
            emit("\":")?;

            // value
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    let value = settings.get_field::<EnumValue>(&descriptor.id).unwrap().0;

                    write!(&mut fmt_buf, "{value}")?;
                    emit(&fmt_buf)?;
                    fmt_buf.clear();
                }
                SettingKind::Percentage { .. } | SettingKind::FloatRange { .. } => {
                    let value = settings.get_field::<f32>(&descriptor.id).unwrap();
                    if !value.is_finite() {
                        return Err(SerializeSettingsError::InvalidFloat {
                            key: descriptor.id.name,
                            value,
                        });
                    }
                    write!(&mut fmt_buf, "{value}")?;
                    emit(&fmt_buf)?;
                    fmt_buf.clear();
                }
                SettingKind::IntRange { .. } => {
                    let value = settings.get_field::<i32>(&descriptor.id).unwrap();
                    write!(&mut fmt_buf, "{value}")?;
                    emit(&fmt_buf)?;
                    fmt_buf.clear();
                }
                SettingKind::Boolean | SettingKind::Group { .. } => {
                    let value = settings.get_field::<bool>(&descriptor.id).unwrap();
                    write!(&mut fmt_buf, "{value}")?;
                    emit(&fmt_buf)?;
                    fmt_buf.clear();
                }
            }
            emit(",")?;
        }

        // version + trailing bracket
        emit("\"version\":1}")?;

        Ok(())
    }

    pub fn write_json_to_fmt(
        &self,
        settings: &T,
        mut dest: impl core::fmt::Write,
    ) -> Result<(), SerializeSettingsError> {
        self.stream_json(settings, |fragment| {
            dest.write_str(fragment)?;
            Ok(())
        })
    }

    #[cfg(feature = "std")]
    pub fn write_json_to_io(
        &self,
        settings: &T,
        mut dest: impl std::io::Write,
    ) -> Result<(), SerializeSettingsError> {
        self.stream_json(settings, |fragment| {
            dest.write_all(fragment.as_bytes())?;
            Ok(())
        })
    }

    pub fn to_json_string(&self, settings: &T) -> Result<String, SerializeSettingsError> {
        let mut s = String::new();
        self.write_json_to_fmt(settings, &mut s)?;
        Ok(s)
    }

    /// Recursive method for reading the settings within a given list of descriptors (either top-level or within a
    /// group) from a given JSON map and using them to update the given settings struct.
    pub(super) fn settings_from_json(
        json: &SortedMap<Cow<'_, str>, JsonValue<'_>>,
        descriptors: &[SettingDescriptor<T>],
        settings: &mut T,
    ) -> Result<(), ParseSettingsError> {
        for descriptor in descriptors {
            let key = descriptor.id.name;
            match &descriptor.kind {
                SettingKind::Enumeration { .. } => {
                    json.get_and_expect::<u32>(key)?
                        .map(|n| settings.set_field::<EnumValue>(&descriptor.id, EnumValue(n)))
                        .transpose()?;
                }
                SettingKind::FloatRange { range, .. } => {
                    json.get_and_expect::<f32>(key)?.map(|n| {
                        settings
                            .set_field::<f32>(&descriptor.id, n.clamp(*range.start(), *range.end()))
                    });
                }
                SettingKind::Percentage { .. } => {
                    json.get_and_expect::<f32>(key)?
                        .map(|n| settings.set_field::<f32>(&descriptor.id, n.clamp(0.0, 1.0)));
                }
                SettingKind::IntRange { range, .. } => {
                    json.get_and_expect::<i32>(key)?.map(|n| {
                        settings
                            .set_field::<i32>(&descriptor.id, n.clamp(*range.start(), *range.end()))
                    });
                }
                SettingKind::Boolean => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field::<bool>(&descriptor.id, b));
                }
                SettingKind::Group { children, .. } => {
                    json.get_and_expect::<bool>(key)?
                        .map(|b| settings.set_field::<bool>(&descriptor.id, b));
                    Self::settings_from_json(json, children, settings)?;
                }
            }
        }

        Ok(())
    }

    /// Parse settings from a given string of JSON and return a new settings struct.
    pub fn from_json_generic(&self, json: &str) -> Result<T, ParseSettingsError> {
        let parsed_map = parse_json(json)?;

        let version = parsed_map
            .get_and_expect::<f32>("version")?
            .ok_or_else(|| {
                // Detect if the user is trying to import an ntscQT preset, and display a specific error if so
                if parsed_map.contains_key("_composite_preemphasis") {
                    ParseSettingsError::WrongApplication
                } else {
                    ParseSettingsError::MissingField { field: "version" }
                }
            })?;
        if version != 1.0 {
            return Err(ParseSettingsError::UnsupportedVersion { version });
        }

        let mut dst_settings = T::legacy_value();
        Self::settings_from_json(&parsed_map, &self.setting_descriptors, &mut dst_settings)?;

        Ok(dst_settings)
    }

    pub fn all_descriptors(&self) -> SettingDescriptors<'_, T> {
        SettingDescriptors::new(self)
    }
}

/// Iterator over all setting descriptors (nested or not) within a given settings list in depth-first order.
pub struct SettingDescriptors<'a, T: Settings> {
    path: Vec<(&'a [SettingDescriptor<T>], usize)>,
}

impl<'a, T: Settings> Iterator for SettingDescriptors<'a, T> {
    type Item = &'a SettingDescriptor<T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (leaf, index) = self.path.last_mut()?;

            let setting = leaf.get(*index);
            match setting {
                Some(desc) => {
                    *index += 1;
                    // Increment the index of the *current* path node and then recurse into the group. This means that
                    // it'll point to the node after the group once we're finished processing the group.
                    if let SettingKind::Group { children, .. } = &desc.kind {
                        self.path.push((children.as_slice(), 0));
                    }
                    return Some(desc);
                }
                None => {
                    // If the index is pointing one past the end of the list, we traverse upwards (and do so until we
                    // reach the next setting or the end of the top-level list).
                    self.path.pop();
                }
            }
        }
    }
}

impl<'a, T: Settings> SettingDescriptors<'a, T> {
    fn new(settings_list: &'a SettingsList<T>) -> Self {
        Self {
            path: vec![(&settings_list.setting_descriptors, 0)],
        }
    }
}
