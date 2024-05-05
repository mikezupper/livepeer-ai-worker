// Package worker provides primitives to interact with the openapi HTTP API.
//
// Code generated by github.com/deepmap/oapi-codegen/v2 version v2.1.0 DO NOT EDIT.
package worker

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"path"
	"strings"

	"github.com/getkin/kin-openapi/openapi3"
	"github.com/go-chi/chi/v5"
	"github.com/oapi-codegen/runtime"
	openapi_types "github.com/oapi-codegen/runtime/types"
)

const (
	HTTPBearerScopes = "HTTPBearer.Scopes"
)

// APIError defines model for APIError.
type APIError struct {
	Msg string `json:"msg"`
}

// BodyImageToImageImageToImagePost defines model for Body_image_to_image_image_to_image_post.
type BodyImageToImageImageToImagePost struct {
	GuidanceScale      *float32           `json:"guidance_scale,omitempty"`
	Image              openapi_types.File `json:"image"`
	ModelId            *string            `json:"model_id,omitempty"`
	NegativePrompt     *string            `json:"negative_prompt,omitempty"`
	NumImagesPerPrompt *int               `json:"num_images_per_prompt,omitempty"`
	Prompt             string             `json:"prompt"`
	SafetyCheck        *bool              `json:"safety_check,omitempty"`
	Seed               *int               `json:"seed,omitempty"`
	Strength           *float32           `json:"strength,omitempty"`
}

// BodyImageToVideoImageToVideoPost defines model for Body_image_to_video_image_to_video_post.
type BodyImageToVideoImageToVideoPost struct {
	Fps              *int               `json:"fps,omitempty"`
	Height           *int               `json:"height,omitempty"`
	Image            openapi_types.File `json:"image"`
	ModelId          *string            `json:"model_id,omitempty"`
	MotionBucketId   *int               `json:"motion_bucket_id,omitempty"`
	NoiseAugStrength *float32           `json:"noise_aug_strength,omitempty"`
	Seed             *int               `json:"seed,omitempty"`
	Width            *int               `json:"width,omitempty"`
}

// HTTPError defines model for HTTPError.
type HTTPError struct {
	Detail APIError `json:"detail"`
}

// HTTPValidationError defines model for HTTPValidationError.
type HTTPValidationError struct {
	Detail *[]ValidationError `json:"detail,omitempty"`
}

// HealthCheck defines model for HealthCheck.
type HealthCheck struct {
	Status *string `json:"status,omitempty"`
}

// ImageResponse defines model for ImageResponse.
type ImageResponse struct {
	Images []Media `json:"images"`
}

// Media defines model for Media.
type Media struct {
	Nsfw bool   `json:"nsfw"`
	Seed int    `json:"seed"`
	Url  string `json:"url"`
}

// TextToImageParams defines model for TextToImageParams.
type TextToImageParams struct {
	GuidanceScale      *float32 `json:"guidance_scale,omitempty"`
	Height             *int     `json:"height,omitempty"`
	ModelId            *string  `json:"model_id,omitempty"`
	NegativePrompt     *string  `json:"negative_prompt,omitempty"`
	NumImagesPerPrompt *int     `json:"num_images_per_prompt,omitempty"`
	Prompt             string   `json:"prompt"`
	SafetyCheck        *bool    `json:"safety_check,omitempty"`
	Seed               *int     `json:"seed,omitempty"`
	Width              *int     `json:"width,omitempty"`
}

// ValidationError defines model for ValidationError.
type ValidationError struct {
	Loc  []ValidationError_Loc_Item `json:"loc"`
	Msg  string                     `json:"msg"`
	Type string                     `json:"type"`
}

// ValidationErrorLoc0 defines model for .
type ValidationErrorLoc0 = string

// ValidationErrorLoc1 defines model for .
type ValidationErrorLoc1 = int

// ValidationError_Loc_Item defines model for ValidationError.loc.Item.
type ValidationError_Loc_Item struct {
	union json.RawMessage
}

// VideoResponse defines model for VideoResponse.
type VideoResponse struct {
	Frames [][]Media `json:"frames"`
}

// ImageToImageMultipartRequestBody defines body for ImageToImage for multipart/form-data ContentType.
type ImageToImageMultipartRequestBody = BodyImageToImageImageToImagePost

// ImageToVideoMultipartRequestBody defines body for ImageToVideo for multipart/form-data ContentType.
type ImageToVideoMultipartRequestBody = BodyImageToVideoImageToVideoPost

// TextToImageJSONRequestBody defines body for TextToImage for application/json ContentType.
type TextToImageJSONRequestBody = TextToImageParams

// AsValidationErrorLoc0 returns the union data inside the ValidationError_Loc_Item as a ValidationErrorLoc0
func (t ValidationError_Loc_Item) AsValidationErrorLoc0() (ValidationErrorLoc0, error) {
	var body ValidationErrorLoc0
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromValidationErrorLoc0 overwrites any union data inside the ValidationError_Loc_Item as the provided ValidationErrorLoc0
func (t *ValidationError_Loc_Item) FromValidationErrorLoc0(v ValidationErrorLoc0) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeValidationErrorLoc0 performs a merge with any union data inside the ValidationError_Loc_Item, using the provided ValidationErrorLoc0
func (t *ValidationError_Loc_Item) MergeValidationErrorLoc0(v ValidationErrorLoc0) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

// AsValidationErrorLoc1 returns the union data inside the ValidationError_Loc_Item as a ValidationErrorLoc1
func (t ValidationError_Loc_Item) AsValidationErrorLoc1() (ValidationErrorLoc1, error) {
	var body ValidationErrorLoc1
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromValidationErrorLoc1 overwrites any union data inside the ValidationError_Loc_Item as the provided ValidationErrorLoc1
func (t *ValidationError_Loc_Item) FromValidationErrorLoc1(v ValidationErrorLoc1) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// MergeValidationErrorLoc1 performs a merge with any union data inside the ValidationError_Loc_Item, using the provided ValidationErrorLoc1
func (t *ValidationError_Loc_Item) MergeValidationErrorLoc1(v ValidationErrorLoc1) error {
	b, err := json.Marshal(v)
	if err != nil {
		return err
	}

	merged, err := runtime.JSONMerge(t.union, b)
	t.union = merged
	return err
}

func (t ValidationError_Loc_Item) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *ValidationError_Loc_Item) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// RequestEditorFn  is the function signature for the RequestEditor callback function
type RequestEditorFn func(ctx context.Context, req *http.Request) error

// Doer performs HTTP requests.
//
// The standard http.Client implements this interface.
type HttpRequestDoer interface {
	Do(req *http.Request) (*http.Response, error)
}

// Client which conforms to the OpenAPI3 specification for this service.
type Client struct {
	// The endpoint of the server conforming to this interface, with scheme,
	// https://api.deepmap.com for example. This can contain a path relative
	// to the server, such as https://api.deepmap.com/dev-test, and all the
	// paths in the swagger spec will be appended to the server.
	Server string

	// Doer for performing requests, typically a *http.Client with any
	// customized settings, such as certificate chains.
	Client HttpRequestDoer

	// A list of callbacks for modifying requests which are generated before sending over
	// the network.
	RequestEditors []RequestEditorFn
}

// ClientOption allows setting custom parameters during construction
type ClientOption func(*Client) error

// Creates a new Client, with reasonable defaults
func NewClient(server string, opts ...ClientOption) (*Client, error) {
	// create a client with sane default values
	client := Client{
		Server: server,
	}
	// mutate client and add all optional params
	for _, o := range opts {
		if err := o(&client); err != nil {
			return nil, err
		}
	}
	// ensure the server URL always has a trailing slash
	if !strings.HasSuffix(client.Server, "/") {
		client.Server += "/"
	}
	// create httpClient, if not already present
	if client.Client == nil {
		client.Client = &http.Client{}
	}
	return &client, nil
}

// WithHTTPClient allows overriding the default Doer, which is
// automatically created using http.Client. This is useful for tests.
func WithHTTPClient(doer HttpRequestDoer) ClientOption {
	return func(c *Client) error {
		c.Client = doer
		return nil
	}
}

// WithRequestEditorFn allows setting up a callback function, which will be
// called right before sending the request. This can be used to mutate the request.
func WithRequestEditorFn(fn RequestEditorFn) ClientOption {
	return func(c *Client) error {
		c.RequestEditors = append(c.RequestEditors, fn)
		return nil
	}
}

// The interface specification for the client above.
type ClientInterface interface {
	// Health request
	Health(ctx context.Context, reqEditors ...RequestEditorFn) (*http.Response, error)

	// ImageToImageWithBody request with any body
	ImageToImageWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error)

	// ImageToVideoWithBody request with any body
	ImageToVideoWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error)

	// TextToImageWithBody request with any body
	TextToImageWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error)

	TextToImage(ctx context.Context, body TextToImageJSONRequestBody, reqEditors ...RequestEditorFn) (*http.Response, error)
}

func (c *Client) Health(ctx context.Context, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewHealthRequest(c.Server)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) ImageToImageWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewImageToImageRequestWithBody(c.Server, contentType, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) ImageToVideoWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewImageToVideoRequestWithBody(c.Server, contentType, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) TextToImageWithBody(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewTextToImageRequestWithBody(c.Server, contentType, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

func (c *Client) TextToImage(ctx context.Context, body TextToImageJSONRequestBody, reqEditors ...RequestEditorFn) (*http.Response, error) {
	req, err := NewTextToImageRequest(c.Server, body)
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	if err := c.applyEditors(ctx, req, reqEditors); err != nil {
		return nil, err
	}
	return c.Client.Do(req)
}

// NewHealthRequest generates requests for Health
func NewHealthRequest(server string) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/health")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("GET", queryURL.String(), nil)
	if err != nil {
		return nil, err
	}

	return req, nil
}

// NewImageToImageRequestWithBody generates requests for ImageToImage with any type of body
func NewImageToImageRequestWithBody(server string, contentType string, body io.Reader) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/image-to-image")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", queryURL.String(), body)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", contentType)

	return req, nil
}

// NewImageToVideoRequestWithBody generates requests for ImageToVideo with any type of body
func NewImageToVideoRequestWithBody(server string, contentType string, body io.Reader) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/image-to-video")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", queryURL.String(), body)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", contentType)

	return req, nil
}

// NewTextToImageRequest calls the generic TextToImage builder with application/json body
func NewTextToImageRequest(server string, body TextToImageJSONRequestBody) (*http.Request, error) {
	var bodyReader io.Reader
	buf, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	bodyReader = bytes.NewReader(buf)
	return NewTextToImageRequestWithBody(server, "application/json", bodyReader)
}

// NewTextToImageRequestWithBody generates requests for TextToImage with any type of body
func NewTextToImageRequestWithBody(server string, contentType string, body io.Reader) (*http.Request, error) {
	var err error

	serverURL, err := url.Parse(server)
	if err != nil {
		return nil, err
	}

	operationPath := fmt.Sprintf("/text-to-image")
	if operationPath[0] == '/' {
		operationPath = "." + operationPath
	}

	queryURL, err := serverURL.Parse(operationPath)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest("POST", queryURL.String(), body)
	if err != nil {
		return nil, err
	}

	req.Header.Add("Content-Type", contentType)

	return req, nil
}

func (c *Client) applyEditors(ctx context.Context, req *http.Request, additionalEditors []RequestEditorFn) error {
	for _, r := range c.RequestEditors {
		if err := r(ctx, req); err != nil {
			return err
		}
	}
	for _, r := range additionalEditors {
		if err := r(ctx, req); err != nil {
			return err
		}
	}
	return nil
}

// ClientWithResponses builds on ClientInterface to offer response payloads
type ClientWithResponses struct {
	ClientInterface
}

// NewClientWithResponses creates a new ClientWithResponses, which wraps
// Client with return type handling
func NewClientWithResponses(server string, opts ...ClientOption) (*ClientWithResponses, error) {
	client, err := NewClient(server, opts...)
	if err != nil {
		return nil, err
	}
	return &ClientWithResponses{client}, nil
}

// WithBaseURL overrides the baseURL.
func WithBaseURL(baseURL string) ClientOption {
	return func(c *Client) error {
		newBaseURL, err := url.Parse(baseURL)
		if err != nil {
			return err
		}
		c.Server = newBaseURL.String()
		return nil
	}
}

// ClientWithResponsesInterface is the interface specification for the client with responses above.
type ClientWithResponsesInterface interface {
	// HealthWithResponse request
	HealthWithResponse(ctx context.Context, reqEditors ...RequestEditorFn) (*HealthResponse, error)

	// ImageToImageWithBodyWithResponse request with any body
	ImageToImageWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*ImageToImageResponse, error)

	// ImageToVideoWithBodyWithResponse request with any body
	ImageToVideoWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*ImageToVideoResponse, error)

	// TextToImageWithBodyWithResponse request with any body
	TextToImageWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*TextToImageResponse, error)

	TextToImageWithResponse(ctx context.Context, body TextToImageJSONRequestBody, reqEditors ...RequestEditorFn) (*TextToImageResponse, error)
}

type HealthResponse struct {
	Body         []byte
	HTTPResponse *http.Response
	JSON200      *HealthCheck
}

// Status returns HTTPResponse.Status
func (r HealthResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r HealthResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

type ImageToImageResponse struct {
	Body         []byte
	HTTPResponse *http.Response
	JSON200      *ImageResponse
	JSON400      *HTTPError
	JSON422      *HTTPValidationError
	JSON500      *HTTPError
}

// Status returns HTTPResponse.Status
func (r ImageToImageResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r ImageToImageResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

type ImageToVideoResponse struct {
	Body         []byte
	HTTPResponse *http.Response
	JSON200      *VideoResponse
	JSON400      *HTTPError
	JSON422      *HTTPValidationError
	JSON500      *HTTPError
}

// Status returns HTTPResponse.Status
func (r ImageToVideoResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r ImageToVideoResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

type TextToImageResponse struct {
	Body         []byte
	HTTPResponse *http.Response
	JSON200      *ImageResponse
	JSON400      *HTTPError
	JSON422      *HTTPValidationError
	JSON500      *HTTPError
}

// Status returns HTTPResponse.Status
func (r TextToImageResponse) Status() string {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.Status
	}
	return http.StatusText(0)
}

// StatusCode returns HTTPResponse.StatusCode
func (r TextToImageResponse) StatusCode() int {
	if r.HTTPResponse != nil {
		return r.HTTPResponse.StatusCode
	}
	return 0
}

// HealthWithResponse request returning *HealthResponse
func (c *ClientWithResponses) HealthWithResponse(ctx context.Context, reqEditors ...RequestEditorFn) (*HealthResponse, error) {
	rsp, err := c.Health(ctx, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseHealthResponse(rsp)
}

// ImageToImageWithBodyWithResponse request with arbitrary body returning *ImageToImageResponse
func (c *ClientWithResponses) ImageToImageWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*ImageToImageResponse, error) {
	rsp, err := c.ImageToImageWithBody(ctx, contentType, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseImageToImageResponse(rsp)
}

// ImageToVideoWithBodyWithResponse request with arbitrary body returning *ImageToVideoResponse
func (c *ClientWithResponses) ImageToVideoWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*ImageToVideoResponse, error) {
	rsp, err := c.ImageToVideoWithBody(ctx, contentType, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseImageToVideoResponse(rsp)
}

// TextToImageWithBodyWithResponse request with arbitrary body returning *TextToImageResponse
func (c *ClientWithResponses) TextToImageWithBodyWithResponse(ctx context.Context, contentType string, body io.Reader, reqEditors ...RequestEditorFn) (*TextToImageResponse, error) {
	rsp, err := c.TextToImageWithBody(ctx, contentType, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseTextToImageResponse(rsp)
}

func (c *ClientWithResponses) TextToImageWithResponse(ctx context.Context, body TextToImageJSONRequestBody, reqEditors ...RequestEditorFn) (*TextToImageResponse, error) {
	rsp, err := c.TextToImage(ctx, body, reqEditors...)
	if err != nil {
		return nil, err
	}
	return ParseTextToImageResponse(rsp)
}

// ParseHealthResponse parses an HTTP response from a HealthWithResponse call
func ParseHealthResponse(rsp *http.Response) (*HealthResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &HealthResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	switch {
	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 200:
		var dest HealthCheck
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON200 = &dest

	}

	return response, nil
}

// ParseImageToImageResponse parses an HTTP response from a ImageToImageWithResponse call
func ParseImageToImageResponse(rsp *http.Response) (*ImageToImageResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &ImageToImageResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	switch {
	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 200:
		var dest ImageResponse
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON200 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 400:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON400 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 422:
		var dest HTTPValidationError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON422 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 500:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON500 = &dest

	}

	return response, nil
}

// ParseImageToVideoResponse parses an HTTP response from a ImageToVideoWithResponse call
func ParseImageToVideoResponse(rsp *http.Response) (*ImageToVideoResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &ImageToVideoResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	switch {
	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 200:
		var dest VideoResponse
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON200 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 400:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON400 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 422:
		var dest HTTPValidationError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON422 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 500:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON500 = &dest

	}

	return response, nil
}

// ParseTextToImageResponse parses an HTTP response from a TextToImageWithResponse call
func ParseTextToImageResponse(rsp *http.Response) (*TextToImageResponse, error) {
	bodyBytes, err := io.ReadAll(rsp.Body)
	defer func() { _ = rsp.Body.Close() }()
	if err != nil {
		return nil, err
	}

	response := &TextToImageResponse{
		Body:         bodyBytes,
		HTTPResponse: rsp,
	}

	switch {
	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 200:
		var dest ImageResponse
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON200 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 400:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON400 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 422:
		var dest HTTPValidationError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON422 = &dest

	case strings.Contains(rsp.Header.Get("Content-Type"), "json") && rsp.StatusCode == 500:
		var dest HTTPError
		if err := json.Unmarshal(bodyBytes, &dest); err != nil {
			return nil, err
		}
		response.JSON500 = &dest

	}

	return response, nil
}

// ServerInterface represents all server handlers.
type ServerInterface interface {
	// Health
	// (GET /health)
	Health(w http.ResponseWriter, r *http.Request)
	// Image To Image
	// (POST /image-to-image)
	ImageToImage(w http.ResponseWriter, r *http.Request)
	// Image To Video
	// (POST /image-to-video)
	ImageToVideo(w http.ResponseWriter, r *http.Request)
	// Text To Image
	// (POST /text-to-image)
	TextToImage(w http.ResponseWriter, r *http.Request)
}

// Unimplemented server implementation that returns http.StatusNotImplemented for each endpoint.

type Unimplemented struct{}

// Health
// (GET /health)
func (_ Unimplemented) Health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
}

// Image To Image
// (POST /image-to-image)
func (_ Unimplemented) ImageToImage(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
}

// Image To Video
// (POST /image-to-video)
func (_ Unimplemented) ImageToVideo(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
}

// Text To Image
// (POST /text-to-image)
func (_ Unimplemented) TextToImage(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusNotImplemented)
}

// ServerInterfaceWrapper converts contexts to parameters.
type ServerInterfaceWrapper struct {
	Handler            ServerInterface
	HandlerMiddlewares []MiddlewareFunc
	ErrorHandlerFunc   func(w http.ResponseWriter, r *http.Request, err error)
}

type MiddlewareFunc func(http.Handler) http.Handler

// Health operation middleware
func (siw *ServerInterfaceWrapper) Health(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		siw.Handler.Health(w, r)
	}))

	for _, middleware := range siw.HandlerMiddlewares {
		handler = middleware(handler)
	}

	handler.ServeHTTP(w, r.WithContext(ctx))
}

// ImageToImage operation middleware
func (siw *ServerInterfaceWrapper) ImageToImage(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	ctx = context.WithValue(ctx, HTTPBearerScopes, []string{})

	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		siw.Handler.ImageToImage(w, r)
	}))

	for _, middleware := range siw.HandlerMiddlewares {
		handler = middleware(handler)
	}

	handler.ServeHTTP(w, r.WithContext(ctx))
}

// ImageToVideo operation middleware
func (siw *ServerInterfaceWrapper) ImageToVideo(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	ctx = context.WithValue(ctx, HTTPBearerScopes, []string{})

	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		siw.Handler.ImageToVideo(w, r)
	}))

	for _, middleware := range siw.HandlerMiddlewares {
		handler = middleware(handler)
	}

	handler.ServeHTTP(w, r.WithContext(ctx))
}

// TextToImage operation middleware
func (siw *ServerInterfaceWrapper) TextToImage(w http.ResponseWriter, r *http.Request) {
	ctx := r.Context()

	ctx = context.WithValue(ctx, HTTPBearerScopes, []string{})

	handler := http.Handler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		siw.Handler.TextToImage(w, r)
	}))

	for _, middleware := range siw.HandlerMiddlewares {
		handler = middleware(handler)
	}

	handler.ServeHTTP(w, r.WithContext(ctx))
}

type UnescapedCookieParamError struct {
	ParamName string
	Err       error
}

func (e *UnescapedCookieParamError) Error() string {
	return fmt.Sprintf("error unescaping cookie parameter '%s'", e.ParamName)
}

func (e *UnescapedCookieParamError) Unwrap() error {
	return e.Err
}

type UnmarshalingParamError struct {
	ParamName string
	Err       error
}

func (e *UnmarshalingParamError) Error() string {
	return fmt.Sprintf("Error unmarshaling parameter %s as JSON: %s", e.ParamName, e.Err.Error())
}

func (e *UnmarshalingParamError) Unwrap() error {
	return e.Err
}

type RequiredParamError struct {
	ParamName string
}

func (e *RequiredParamError) Error() string {
	return fmt.Sprintf("Query argument %s is required, but not found", e.ParamName)
}

type RequiredHeaderError struct {
	ParamName string
	Err       error
}

func (e *RequiredHeaderError) Error() string {
	return fmt.Sprintf("Header parameter %s is required, but not found", e.ParamName)
}

func (e *RequiredHeaderError) Unwrap() error {
	return e.Err
}

type InvalidParamFormatError struct {
	ParamName string
	Err       error
}

func (e *InvalidParamFormatError) Error() string {
	return fmt.Sprintf("Invalid format for parameter %s: %s", e.ParamName, e.Err.Error())
}

func (e *InvalidParamFormatError) Unwrap() error {
	return e.Err
}

type TooManyValuesForParamError struct {
	ParamName string
	Count     int
}

func (e *TooManyValuesForParamError) Error() string {
	return fmt.Sprintf("Expected one value for %s, got %d", e.ParamName, e.Count)
}

// Handler creates http.Handler with routing matching OpenAPI spec.
func Handler(si ServerInterface) http.Handler {
	return HandlerWithOptions(si, ChiServerOptions{})
}

type ChiServerOptions struct {
	BaseURL          string
	BaseRouter       chi.Router
	Middlewares      []MiddlewareFunc
	ErrorHandlerFunc func(w http.ResponseWriter, r *http.Request, err error)
}

// HandlerFromMux creates http.Handler with routing matching OpenAPI spec based on the provided mux.
func HandlerFromMux(si ServerInterface, r chi.Router) http.Handler {
	return HandlerWithOptions(si, ChiServerOptions{
		BaseRouter: r,
	})
}

func HandlerFromMuxWithBaseURL(si ServerInterface, r chi.Router, baseURL string) http.Handler {
	return HandlerWithOptions(si, ChiServerOptions{
		BaseURL:    baseURL,
		BaseRouter: r,
	})
}

// HandlerWithOptions creates http.Handler with additional options
func HandlerWithOptions(si ServerInterface, options ChiServerOptions) http.Handler {
	r := options.BaseRouter

	if r == nil {
		r = chi.NewRouter()
	}
	if options.ErrorHandlerFunc == nil {
		options.ErrorHandlerFunc = func(w http.ResponseWriter, r *http.Request, err error) {
			http.Error(w, err.Error(), http.StatusBadRequest)
		}
	}
	wrapper := ServerInterfaceWrapper{
		Handler:            si,
		HandlerMiddlewares: options.Middlewares,
		ErrorHandlerFunc:   options.ErrorHandlerFunc,
	}

	r.Group(func(r chi.Router) {
		r.Get(options.BaseURL+"/health", wrapper.Health)
	})
	r.Group(func(r chi.Router) {
		r.Post(options.BaseURL+"/image-to-image", wrapper.ImageToImage)
	})
	r.Group(func(r chi.Router) {
		r.Post(options.BaseURL+"/image-to-video", wrapper.ImageToVideo)
	})
	r.Group(func(r chi.Router) {
		r.Post(options.BaseURL+"/text-to-image", wrapper.TextToImage)
	})

	return r
}

// Base64 encoded, gzipped, json marshaled Swagger object
var swaggerSpec = []string{

	"H4sIAAAAAAAC/+xX3W/bNhD/Vwhuj47teM0y+C3JttbY0gax1z0EgcFIZ5mtRHL8SGoE/t8HHm2J+vDs",
	"bGkKDHmyJd3H7453vzs+0kQWSgoQ1tDxIzXJEgqGf8+uJr9oLbX/r7RUoC0H/FKYzP9YbnOgY3ppMtqj",
	"dqX8g7Gai4yu1z2q4S/HNaR0fIMqt71SpbRd6sm7T5BYuu7Rc5mu5rxgGcyt3PxpPCppbBtW5njKRAJz",
	"kzDv5ZGmsGAut3R82j+pnL/dyJEpypUQhCvuQHsI6MUbWEhdMEvH9I4Lple0MjJBkVbYPVrIFPI5T2v+",
	"aaR56QXIJO1SFpAxy+9hrrQslN1p4/1GjlwFuS5TrgjZMnMFusvgcWTPFQQjMuQKdMsqFxaykJrKzlZ3",
	"NwTDFmBX82QJyeea5wXLDVTepyhHLlCutHMnZQ5MoCGANHY59c9d6IzVIDK7rHkb9n+KfG0lWsfeqFi1",
	"DSsUQ1S8hxbo3tq+5ynI5mN3bS+UqcX0YwXnV2U6c7EEni3rJ35yGum9C9+7VL9Z/RfScinmdy75DLZp",
	"5Hh0GlvxkuQcJWvWojiE5AbmzGXzHYUxHEU94IXJmcvI7hp5Qik+8LTh7ng4elO5+xO/tzUbZbin+naX",
	"UEf1vZvNrnZQegqW8dz/+17Dgo7pd4NqMAw2U2FQ0nYT5UY9gln52gHkI8t5yvwh7oXELRRmH7amvXWF",
	"5edgqQTCtGYrjCFG2zTQhRtYbpcXWz6r4zWWWVfvUvrhNxpTDwp0jcqqJysHHf6x6a7BKCkMtBEEuj84",
	"Y5eQchbnKUyArjy1KtLEZ12H1YE7eGrhFWbxEPfSe//8n/jf6TyW+0PnezcThzImWEREUWQBeEdEM/hi",
	"ZxIDv2KahWR/rXWkYvIDuPt1//iG+0dJ+k9k+U1UUeW1C6yjCvdyaC6TGh0wsfqwoOObx1ayHlsQbyNm",
	"+F0m6KbFDb3WZQCM2bEZhBeVKGImM/92X5P6OIKrjWSUqQN4+6Mfi7t5c6FZ0eDNJxJoIyflahYM7yHU",
	"jfs4pBreVkBYkYnT3K6mHkrA7mfYOTANurzIYRmHV6WRpbWKrr0NLhYytIVJNFd4vmN6JghTKufhwImV",
	"RDtBziZEcQU5FyGebV3we1AA2n+/dkKgo3vQJtga9o/7Q58QqUAwxemY/oCvelQxu0TYgyXOPGRMwMb2",
	"R4POJ2k5EqlPWcgHao2GQ/+TSGFBoFYEevDJePfb2+y+Y4yHLiamnpCpSxIwZuFyUh4JHoErCr8TlxD9",
	"ywFS3pGVR+UOvV3o62FhZ28anIZ6AGP9cteIq3C55YppO/DL+FHKLDs8tEOvKut6TVrtYP0VM15fGA7N",
	"eY++ec5TLxfUDv/nLCXX4UjQ72j0rH5bu2obQSVCyn325KXCnwgLWrCcTEHfgybV0r/lHZwhMePc3K5v",
	"457AIyYzGcZ6ozfwmrK3N5AFX6o3dl+kXrg36tz/2hv/594IFY69YeGLPWBsRGvhP3bGvw++vXi+DofX",
	"BnjeBvA1Fs8G1PXGDKrW/ZU75kUuXUouZFE4we2KvGUWHtiKbq79uNma8WCQamDFURa+9vONej/x6v5a",
	"83cAAAD//4J+tev0GAAA",
}

// GetSwagger returns the content of the embedded swagger specification file
// or error if failed to decode
func decodeSpec() ([]byte, error) {
	zipped, err := base64.StdEncoding.DecodeString(strings.Join(swaggerSpec, ""))
	if err != nil {
		return nil, fmt.Errorf("error base64 decoding spec: %w", err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(zipped))
	if err != nil {
		return nil, fmt.Errorf("error decompressing spec: %w", err)
	}
	var buf bytes.Buffer
	_, err = buf.ReadFrom(zr)
	if err != nil {
		return nil, fmt.Errorf("error decompressing spec: %w", err)
	}

	return buf.Bytes(), nil
}

var rawSpec = decodeSpecCached()

// a naive cached of a decoded swagger spec
func decodeSpecCached() func() ([]byte, error) {
	data, err := decodeSpec()
	return func() ([]byte, error) {
		return data, err
	}
}

// Constructs a synthetic filesystem for resolving external references when loading openapi specifications.
func PathToRawSpec(pathToFile string) map[string]func() ([]byte, error) {
	res := make(map[string]func() ([]byte, error))
	if len(pathToFile) > 0 {
		res[pathToFile] = rawSpec
	}

	return res
}

// GetSwagger returns the Swagger specification corresponding to the generated code
// in this file. The external references of Swagger specification are resolved.
// The logic of resolving external references is tightly connected to "import-mapping" feature.
// Externally referenced files must be embedded in the corresponding golang packages.
// Urls can be supported but this task was out of the scope.
func GetSwagger() (swagger *openapi3.T, err error) {
	resolvePath := PathToRawSpec("")

	loader := openapi3.NewLoader()
	loader.IsExternalRefsAllowed = true
	loader.ReadFromURIFunc = func(loader *openapi3.Loader, url *url.URL) ([]byte, error) {
		pathToFile := url.String()
		pathToFile = path.Clean(pathToFile)
		getSpec, ok := resolvePath[pathToFile]
		if !ok {
			err1 := fmt.Errorf("path not found: %s", pathToFile)
			return nil, err1
		}
		return getSpec()
	}
	var specData []byte
	specData, err = rawSpec()
	if err != nil {
		return
	}
	swagger, err = loader.LoadFromData(specData)
	if err != nil {
		return
	}
	return
}
