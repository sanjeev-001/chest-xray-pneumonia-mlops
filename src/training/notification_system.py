"""
Stakeholder Notification System for Retraining Workflows
Sends notifications about retraining status, results, and alerts
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    from email.mime.base import MimeBase
    from email import encoders
    import smtplib
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    MimeText = None
    MimeMultipart = None
    MimeBase = None
    encoders = None
    smtplib = None
import threading
import time
import uuid
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False

try:
    from .retraining_system import RetrainingResult, RetrainingStatus, RetrainingTrigger
except ImportError:
    from retraining_system import RetrainingResult, RetrainingStatus, RetrainingTrigger

logger = logging.getLogger(__name__)

class NotificationType(Enum):
    """Types of notifications"""
    RETRAINING_STARTED = "retraining_started"
    RETRAINING_COMPLETED = "retraining_completed"
    RETRAINING_FAILED = "retraining_failed"
    MODEL_PROMOTED = "model_promoted"
    MODEL_NOT_PROMOTED = "model_not_promoted"
    PERFORMANCE_ALERT = "performance_alert"
    DRIFT_ALERT = "drift_alert"
    SYSTEM_ERROR = "system_error"

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"

@dataclass
class NotificationRecipient:
    """Notification recipient configuration"""
    name: str
    email: Optional[str] = None
    slack_user_id: Optional[str] = None
    phone: Optional[str] = None
    notification_types: List[NotificationType] = None
    channels: List[NotificationChannel] = None
    priority_threshold: NotificationPriority = NotificationPriority.MEDIUM
    
    def __post_init__(self):
        if self.notification_types is None:
            self.notification_types = list(NotificationType)
        if self.channels is None:
            self.channels = [NotificationChannel.EMAIL]

@dataclass
class NotificationMessage:
    """Notification message structure"""
    notification_id: str
    notification_type: NotificationType
    priority: NotificationPriority
    subject: str
    message: str
    data: Dict[str, Any]
    recipients: List[str]
    channels: List[NotificationChannel]
    created_at: datetime
    sent_at: Optional[datetime] = None
    delivery_status: Dict[str, str] = None
    
    def __post_init__(self):
        if not self.notification_id:
            self.notification_id = str(uuid.uuid4())
        if self.delivery_status is None:
            self.delivery_status = {}

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, 
                 smtp_host: str,
                 smtp_port: int = 587,
                 username: str = None,
                 password: str = None,
                 from_email: str = None,
                 use_tls: bool = True):
        
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email or username
        self.use_tls = use_tls
        
        logger.info(f"EmailNotifier initialized: {smtp_host}:{smtp_port}")
    
    def send_notification(self, message: NotificationMessage, recipients: List[NotificationRecipient]) -> Dict[str, str]:
        """Send email notification"""
        if not EMAIL_AVAILABLE:
            return {"error": "email modules not available"}
        
        results = {}
        
        try:
            # Create email message
            msg = MimeMultipart()
            msg['From'] = self.from_email
            msg['Subject'] = message.subject
            
            # Create HTML body
            html_body = self._create_html_body(message)
            msg.attach(MimeText(html_body, 'html'))
            
            # Add JSON attachment with detailed data
            if message.data:
                json_attachment = MimeBase('application', 'json')
                json_attachment.set_payload(json.dumps(message.data, indent=2, default=str))
                encoders.encode_base64(json_attachment)
                json_attachment.add_header(
                    'Content-Disposition',
                    f'attachment; filename="notification_data_{message.notification_id[:8]}.json"'
                )
                msg.attach(json_attachment)
            
            # Send to each recipient
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                if self.username and self.password:
                    server.login(self.username, self.password)
                
                for recipient in recipients:
                    if recipient.email:
                        try:
                            msg['To'] = recipient.email
                            server.send_message(msg)
                            results[recipient.email] = "sent"
                            logger.info(f"Email sent to {recipient.email}")
                        except Exception as e:
                            results[recipient.email] = f"failed: {str(e)}"
                            logger.error(f"Failed to send email to {recipient.email}: {e}")
                        finally:
                            if 'To' in msg:
                                del msg['To']
        
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            for recipient in recipients:
                if recipient.email:
                    results[recipient.email] = f"failed: {str(e)}"
        
        return results
    
    def _create_html_body(self, message: NotificationMessage) -> str:
        """Create HTML email body"""
        
        # Priority color mapping
        priority_colors = {
            NotificationPriority.LOW: "#28a745",
            NotificationPriority.MEDIUM: "#ffc107", 
            NotificationPriority.HIGH: "#fd7e14",
            NotificationPriority.CRITICAL: "#dc3545"
        }
        
        color = priority_colors.get(message.priority, "#6c757d")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 5px; }}
                .content {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .footer {{ margin-top: 20px; font-size: 12px; color: #666; }}
                .data-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
                .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .data-table th {{ background-color: #f2f2f2; }}
                .priority-{message.priority.value} {{ color: {color}; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🔔 MLOps Notification</h2>
                <p><strong>Type:</strong> {message.notification_type.value.replace('_', ' ').title()}</p>
                <p><strong>Priority:</strong> <span class="priority-{message.priority.value}">{message.priority.value.upper()}</span></p>
                <p><strong>Time:</strong> {message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
            
            <div class="content">
                <h3>Message</h3>
                <p>{message.message.replace(chr(10), '<br>')}</p>
                
                {self._format_data_section(message.data)}
            </div>
            
            <div class="footer">
                <p>This is an automated notification from the Chest X-Ray Pneumonia Detection MLOps System.</p>
                <p>Notification ID: {message.notification_id}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_data_section(self, data: Dict[str, Any]) -> str:
        """Format data section for email"""
        if not data:
            return ""
        
        html = "<h3>Details</h3><table class='data-table'>"
        
        for key, value in data.items():
            if isinstance(value, dict):
                value_str = json.dumps(value, indent=2)
                html += f"<tr><th>{key.replace('_', ' ').title()}</th><td><pre>{value_str}</pre></td></tr>"
            elif isinstance(value, list):
                value_str = "<br>".join([str(item) for item in value])
                html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value_str}</td></tr>"
            else:
                html += f"<tr><th>{key.replace('_', ' ').title()}</th><td>{value}</td></tr>"
        
        html += "</table>"
        return html

class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, webhook_url: str = None, bot_token: str = None):
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, Slack notifications disabled")
        
        logger.info("SlackNotifier initialized")
    
    def send_notification(self, message: NotificationMessage, recipients: List[NotificationRecipient]) -> Dict[str, str]:
        """Send Slack notification"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests library not available"}
        
        results = {}
        
        try:
            # Create Slack message
            slack_message = self._create_slack_message(message)
            
            if self.webhook_url:
                # Send via webhook
                response = requests.post(
                    self.webhook_url,
                    json=slack_message,
                    timeout=10
                )
                response.raise_for_status()
                results["webhook"] = "sent"
                logger.info("Slack webhook notification sent")
            
            # Send direct messages if bot token is available
            if self.bot_token:
                for recipient in recipients:
                    if recipient.slack_user_id:
                        try:
                            self._send_direct_message(recipient.slack_user_id, slack_message)
                            results[recipient.slack_user_id] = "sent"
                        except Exception as e:
                            results[recipient.slack_user_id] = f"failed: {str(e)}"
        
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def _create_slack_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create Slack message format"""
        
        # Priority emoji mapping
        priority_emojis = {
            NotificationPriority.LOW: "🟢",
            NotificationPriority.MEDIUM: "🟡",
            NotificationPriority.HIGH: "🟠",
            NotificationPriority.CRITICAL: "🔴"
        }
        
        emoji = priority_emojis.get(message.priority, "ℹ️")
        
        # Create attachment with details
        attachment = {
            "color": self._get_slack_color(message.priority),
            "title": f"{emoji} {message.notification_type.value.replace('_', ' ').title()}",
            "text": message.message,
            "fields": [
                {
                    "title": "Priority",
                    "value": message.priority.value.upper(),
                    "short": True
                },
                {
                    "title": "Time",
                    "value": message.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                    "short": True
                }
            ],
            "footer": "MLOps Notification System",
            "ts": int(message.created_at.timestamp())
        }
        
        # Add data fields
        if message.data:
            for key, value in list(message.data.items())[:5]:  # Limit to 5 fields
                if not isinstance(value, (dict, list)):
                    attachment["fields"].append({
                        "title": key.replace('_', ' ').title(),
                        "value": str(value),
                        "short": True
                    })
        
        return {
            "text": f"MLOps Notification: {message.subject}",
            "attachments": [attachment]
        }
    
    def _get_slack_color(self, priority: NotificationPriority) -> str:
        """Get Slack color for priority"""
        colors = {
            NotificationPriority.LOW: "good",
            NotificationPriority.MEDIUM: "warning",
            NotificationPriority.HIGH: "#ff9500",
            NotificationPriority.CRITICAL: "danger"
        }
        return colors.get(priority, "#36a64f")
    
    def _send_direct_message(self, user_id: str, message: Dict[str, Any]):
        """Send direct message to Slack user"""
        if not self.bot_token:
            return
        
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "channel": user_id,
            **message
        }
        
        response = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

class WebhookNotifier:
    """Generic webhook notification handler"""
    
    def __init__(self, webhook_urls: List[str], headers: Dict[str, str] = None):
        self.webhook_urls = webhook_urls
        self.headers = headers or {"Content-Type": "application/json"}
        
        if not REQUESTS_AVAILABLE:
            logger.warning("requests library not available, webhook notifications disabled")
        
        logger.info(f"WebhookNotifier initialized with {len(webhook_urls)} URLs")
    
    def send_notification(self, message: NotificationMessage, recipients: List[NotificationRecipient]) -> Dict[str, str]:
        """Send webhook notification"""
        if not REQUESTS_AVAILABLE:
            return {"error": "requests library not available"}
        
        results = {}
        
        # Create webhook payload
        payload = {
            "notification_id": message.notification_id,
            "type": message.notification_type.value,
            "priority": message.priority.value,
            "subject": message.subject,
            "message": message.message,
            "data": message.data,
            "timestamp": message.created_at.isoformat(),
            "recipients": [r.name for r in recipients]
        }
        
        for i, url in enumerate(self.webhook_urls):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
                results[f"webhook_{i}"] = "sent"
                logger.info(f"Webhook notification sent to {url}")
                
            except Exception as e:
                results[f"webhook_{i}"] = f"failed: {str(e)}"
                logger.error(f"Webhook notification failed for {url}: {e}")
        
        return results

class NotificationManager:
    """Main notification management system"""
    
    def __init__(self):
        self.recipients: Dict[str, NotificationRecipient] = {}
        self.notifiers: Dict[NotificationChannel, Any] = {}
        self.notification_history: List[NotificationMessage] = []
        self.lock = threading.Lock()
        
        logger.info("NotificationManager initialized")
    
    def add_recipient(self, recipient: NotificationRecipient):
        """Add notification recipient"""
        with self.lock:
            self.recipients[recipient.name] = recipient
        logger.info(f"Added recipient: {recipient.name}")
    
    def add_notifier(self, channel: NotificationChannel, notifier):
        """Add notification channel"""
        self.notifiers[channel] = notifier
        logger.info(f"Added notifier: {channel.value}")
    
    def send_notification(self,
                         notification_type: NotificationType,
                         subject: str,
                         message: str,
                         data: Dict[str, Any] = None,
                         priority: NotificationPriority = NotificationPriority.MEDIUM,
                         recipient_names: List[str] = None) -> str:
        """Send notification to specified recipients"""
        
        # Create notification message
        notification = NotificationMessage(
            notification_id=str(uuid.uuid4()),
            notification_type=notification_type,
            priority=priority,
            subject=subject,
            message=message,
            data=data or {},
            recipients=recipient_names or [],
            channels=[],
            created_at=datetime.now()
        )
        
        # Filter recipients
        target_recipients = []
        with self.lock:
            for name in (recipient_names or self.recipients.keys()):
                if name in self.recipients:
                    recipient = self.recipients[name]
                    
                    # Check if recipient wants this notification type
                    if notification_type in recipient.notification_types:
                        # Check priority threshold
                        priority_levels = {
                            NotificationPriority.LOW: 0,
                            NotificationPriority.MEDIUM: 1,
                            NotificationPriority.HIGH: 2,
                            NotificationPriority.CRITICAL: 3
                        }
                        
                        if priority_levels[priority] >= priority_levels[recipient.priority_threshold]:
                            target_recipients.append(recipient)
        
        if not target_recipients:
            logger.warning(f"No recipients found for notification {notification.notification_id}")
            return notification.notification_id
        
        # Collect all channels needed
        all_channels = set()
        for recipient in target_recipients:
            all_channels.update(recipient.channels)
        
        notification.channels = list(all_channels)
        
        # Send via each channel
        delivery_results = {}
        
        for channel in all_channels:
            if channel in self.notifiers:
                try:
                    # Filter recipients for this channel
                    channel_recipients = [
                        r for r in target_recipients 
                        if channel in r.channels
                    ]
                    
                    if channel_recipients:
                        results = self.notifiers[channel].send_notification(
                            notification, 
                            channel_recipients
                        )
                        delivery_results[channel.value] = results
                        
                except Exception as e:
                    logger.error(f"Failed to send via {channel.value}: {e}")
                    delivery_results[channel.value] = {"error": str(e)}
            else:
                logger.warning(f"No notifier configured for channel: {channel.value}")
        
        # Update notification status
        notification.sent_at = datetime.now()
        notification.delivery_status = delivery_results
        
        # Store in history
        with self.lock:
            self.notification_history.append(notification)
            # Keep only last 1000 notifications
            if len(self.notification_history) > 1000:
                self.notification_history = self.notification_history[-1000:]
        
        logger.info(f"Notification sent: {notification.notification_id} to {len(target_recipients)} recipients")
        
        return notification.notification_id
    
    def send_retraining_notification(self, 
                                   retraining_result: RetrainingResult,
                                   additional_data: Dict[str, Any] = None) -> str:
        """Send retraining-specific notification"""
        
        # Determine notification type and priority
        if retraining_result.status == RetrainingStatus.COMPLETED:
            if retraining_result.promoted:
                notification_type = NotificationType.MODEL_PROMOTED
                priority = NotificationPriority.HIGH
                subject = f"🎉 Model Promoted: {retraining_result.new_model_id}"
                message = f"""
Model retraining completed successfully and the new model has been promoted to production.

New Model: {retraining_result.new_model_id}:{retraining_result.new_model_version}
Performance Improvement: {retraining_result.performance_comparison.get('overall_improvement', 0):.1%}
Duration: {(retraining_result.completed_at - retraining_result.started_at).total_seconds() / 60:.1f} minutes

The new model is now serving predictions in production.
                """.strip()
            else:
                notification_type = NotificationType.MODEL_NOT_PROMOTED
                priority = NotificationPriority.MEDIUM
                subject = f"ℹ️ Model Not Promoted: {retraining_result.new_model_id}"
                message = f"""
Model retraining completed but the new model was not promoted to production.

New Model: {retraining_result.new_model_id}:{retraining_result.new_model_version}
Reason: {retraining_result.performance_comparison.get('promotion_reason', 'Performance did not meet promotion criteria')}
Duration: {(retraining_result.completed_at - retraining_result.started_at).total_seconds() / 60:.1f} minutes

The current production model remains unchanged.
                """.strip()
        
        elif retraining_result.status == RetrainingStatus.FAILED:
            notification_type = NotificationType.RETRAINING_FAILED
            priority = NotificationPriority.CRITICAL
            subject = f"❌ Retraining Failed: {retraining_result.request_id}"
            message = f"""
Model retraining failed and requires immediate attention.

Request ID: {retraining_result.request_id}
Error: {retraining_result.error_message}
Duration: {(retraining_result.completed_at - retraining_result.started_at).total_seconds() / 60:.1f} minutes

Please check the system logs and take corrective action.
            """.strip()
        
        else:
            notification_type = NotificationType.RETRAINING_STARTED
            priority = NotificationPriority.LOW
            subject = f"🔄 Retraining Started: {retraining_result.request_id}"
            message = f"""
Model retraining has been initiated.

Request ID: {retraining_result.request_id}
Status: {retraining_result.status.value}
Started: {retraining_result.started_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

You will receive another notification when retraining completes.
            """.strip()
        
        # Prepare notification data
        notification_data = {
            "request_id": retraining_result.request_id,
            "status": retraining_result.status.value,
            "new_model_id": retraining_result.new_model_id,
            "new_model_version": retraining_result.new_model_version,
            "promoted": retraining_result.promoted,
            "started_at": retraining_result.started_at.isoformat(),
            "completed_at": retraining_result.completed_at.isoformat() if retraining_result.completed_at else None,
            "duration_minutes": (retraining_result.completed_at - retraining_result.started_at).total_seconds() / 60 if retraining_result.completed_at else None,
            "performance_comparison": retraining_result.performance_comparison,
            "error_message": retraining_result.error_message
        }
        
        if additional_data:
            notification_data.update(additional_data)
        
        return self.send_notification(
            notification_type=notification_type,
            subject=subject,
            message=message,
            data=notification_data,
            priority=priority
        )
    
    def get_notification_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notification history"""
        with self.lock:
            recent_notifications = self.notification_history[-limit:]
        
        return [
            {
                "notification_id": n.notification_id,
                "type": n.notification_type.value,
                "priority": n.priority.value,
                "subject": n.subject,
                "recipients": n.recipients,
                "channels": [c.value for c in n.channels],
                "created_at": n.created_at.isoformat(),
                "sent_at": n.sent_at.isoformat() if n.sent_at else None,
                "delivery_status": n.delivery_status
            }
            for n in reversed(recent_notifications)
        ]
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """Get notification statistics"""
        with self.lock:
            total_notifications = len(self.notification_history)
            
            if total_notifications == 0:
                return {
                    "total_notifications": 0,
                    "by_type": {},
                    "by_priority": {},
                    "by_channel": {},
                    "success_rate": 0.0
                }
            
            # Count by type
            by_type = {}
            by_priority = {}
            by_channel = {}
            successful_deliveries = 0
            total_deliveries = 0
            
            for notification in self.notification_history:
                # Count by type
                type_key = notification.notification_type.value
                by_type[type_key] = by_type.get(type_key, 0) + 1
                
                # Count by priority
                priority_key = notification.priority.value
                by_priority[priority_key] = by_priority.get(priority_key, 0) + 1
                
                # Count by channel and calculate success rate
                for channel in notification.channels:
                    channel_key = channel.value
                    by_channel[channel_key] = by_channel.get(channel_key, 0) + 1
                
                # Calculate success rate
                if notification.delivery_status:
                    for channel_results in notification.delivery_status.values():
                        if isinstance(channel_results, dict):
                            for result in channel_results.values():
                                total_deliveries += 1
                                if result == "sent":
                                    successful_deliveries += 1
        
        success_rate = successful_deliveries / total_deliveries if total_deliveries > 0 else 0.0
        
        return {
            "total_notifications": total_notifications,
            "by_type": by_type,
            "by_priority": by_priority,
            "by_channel": by_channel,
            "success_rate": success_rate,
            "total_recipients": len(self.recipients)
        }

# Global notification manager instance
_notification_manager = None

def get_notification_manager() -> NotificationManager:
    """Get global notification manager instance"""
    global _notification_manager
    
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    
    return _notification_manager

# Convenience functions
def send_retraining_notification(retraining_result: RetrainingResult, 
                               additional_data: Dict[str, Any] = None) -> str:
    """Send retraining notification using global manager"""
    manager = get_notification_manager()
    return manager.send_retraining_notification(retraining_result, additional_data)

def send_alert_notification(alert_type: str,
                          message: str,
                          data: Dict[str, Any] = None,
                          priority: NotificationPriority = NotificationPriority.HIGH) -> str:
    """Send alert notification using global manager"""
    manager = get_notification_manager()
    
    notification_type = NotificationType.PERFORMANCE_ALERT
    if "drift" in alert_type.lower():
        notification_type = NotificationType.DRIFT_ALERT
    
    return manager.send_notification(
        notification_type=notification_type,
        subject=f"🚨 Alert: {alert_type}",
        message=message,
        data=data,
        priority=priority
    )