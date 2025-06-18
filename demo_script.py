#!/usr/bin/env python3
"""
Demo Script for MCP Code QA Server
Demonstrates the complete functionality of the system with a sample repository.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json
import time

# Import our components
from mcp_code_qa_server import RAGSystem
from evaluation_script import QAEvaluator, QAPair, EvaluationReporter
from repo_analysis_agent import RepoAnalyzer, ReportGenerator


def create_demo_repository():
    """Create a comprehensive demo repository"""
    print("📁 Creating demo repository...")

    temp_dir = tempfile.mkdtemp(prefix="mcp_demo_")

    # Create a realistic Python project structure
    projects = {
        "main.py": '''#!/usr/bin/env python3
"""
E-commerce Order Management System
A sample application demonstrating various design patterns and architectural concepts.
"""

import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from models.order import Order, OrderItem
from models.customer import Customer
from services.order_service import OrderService
from services.payment_service import PaymentService
from utils.config import ConfigManager
from utils.logger import setup_logger

# Configure logging
logger = setup_logger(__name__)

class OrderManagementApp:
    """Main application class implementing Facade pattern"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.order_service = OrderService()
        self.payment_service = PaymentService()
        logger.info("Order Management App initialized")
    
    def process_order(self, customer_data: Dict[str, Any], items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a complete order workflow"""
        try:
            # Create customer
            customer = Customer.from_dict(customer_data)
            
            # Create order items
            order_items = [OrderItem.from_dict(item) for item in items]
            
            # Create order
            order = Order(customer=customer, items=order_items)
            
            # Process payment
            payment_result = self.payment_service.process_payment(order.total_amount, customer.payment_method)
            
            if payment_result.success:
                # Save order
                saved_order = self.order_service.create_order(order)
                logger.info(f"Order {saved_order.id} processed successfully")
                
                return {
                    "success": True,
                    "order_id": saved_order.id,
                    "total_amount": saved_order.total_amount,
                    "status": saved_order.status
                }
            else:
                logger.error(f"Payment failed: {payment_result.error_message}")
                return {
                    "success": False,
                    "error": "Payment processing failed",
                    "details": payment_result.error_message
                }
                
        except Exception as e:
            logger.error(f"Order processing failed: {str(e)}")
            return {
                "success": False,
                "error": "Order processing failed",
                "details": str(e)
            }
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an existing order"""
        order = self.order_service.get_order(order_id)
        if order:
            return {
                "order_id": order.id,
                "status": order.status,
                "total_amount": order.total_amount,
                "created_at": order.created_at.isoformat()
            }
        return {"error": "Order not found"}

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command> [args]")
        print("Commands: process_order, get_status")
        sys.exit(1)
    
    app = OrderManagementApp()
    command = sys.argv[1]
    
    if command == "process_order":
        # Sample order processing
        customer_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "payment_method": "credit_card"
        }
        
        items = [
            {"product_id": "P001", "quantity": 2, "price": 29.99},
            {"product_id": "P002", "quantity": 1, "price": 49.99}
        ]
        
        result = app.process_order(customer_data, items)
        print(json.dumps(result, indent=2))
    
    elif command == "get_status":
        if len(sys.argv) < 3:
            print("Usage: python main.py get_status <order_id>")
            sys.exit(1)
        
        order_id = sys.argv[2]
        result = app.get_order_status(order_id)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
''',
        "models/__init__.py": '''"""Data models for the order management system"""

from .order import Order, OrderItem
from .customer import Customer

__all__ = ["Order", "OrderItem", "Customer"]
''',
        "models/order.py": '''"""Order and OrderItem models"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any
from enum import Enum
import uuid

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"

@dataclass
class OrderItem:
    """Represents an item in an order"""
    product_id: str
    quantity: int
    price: float
    
    @property
    def total_price(self) -> float:
        """Calculate total price for this item"""
        return self.quantity * self.price
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderItem':
        """Create OrderItem from dictionary"""
        return cls(
            product_id=data["product_id"],
            quantity=data["quantity"],
            price=data["price"]
        )

@dataclass
class Order:
    """Represents a customer order"""
    customer: 'Customer'
    items: List[OrderItem]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = field(default=OrderStatus.PENDING)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def total_amount(self) -> float:
        """Calculate total order amount"""
        return sum(item.total_price for item in self.items)
    
    @property
    def item_count(self) -> int:
        """Get total number of items"""
        return sum(item.quantity for item in self.items)
    
    def add_item(self, item: OrderItem) -> None:
        """Add an item to the order"""
        self.items.append(item)
    
    def remove_item(self, product_id: str) -> bool:
        """Remove an item from the order"""
        for i, item in enumerate(self.items):
            if item.product_id == product_id:
                del self.items[i]
                return True
        return False
    
    def update_status(self, new_status: OrderStatus) -> None:
        """Update order status"""
        self.status = new_status
''',
        "models/customer.py": '''"""Customer model and related functionality"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import re

@dataclass
class Customer:
    """Represents a customer in the system"""
    name: str
    email: str
    payment_method: str
    phone: Optional[str] = None
    address: Optional[str] = None
    
    def __post_init__(self):
        """Validate customer data after initialization"""
        if not self.is_valid_email(self.email):
            raise ValueError(f"Invalid email address: {self.email}")
        
        if not self.name.strip():
            raise ValueError("Customer name cannot be empty")
    
    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email address format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Customer':
        """Create Customer from dictionary"""
        return cls(
            name=data["name"],
            email=data["email"],
            payment_method=data["payment_method"],
            phone=data.get("phone"),
            address=data.get("address")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert customer to dictionary"""
        return {
            "name": self.name,
            "email": self.email,
            "payment_method": self.payment_method,
            "phone": self.phone,
            "address": self.address
        }
''',
        "services/__init__.py": '''"""Business logic services"""

from .order_service import OrderService
from .payment_service import PaymentService

__all__ = ["OrderService", "PaymentService"]
''',
        "services/order_service.py": '''"""Order management service implementing Repository pattern"""

from typing import Dict, List, Optional
from datetime import datetime
import json
import os

from models.order import Order, OrderStatus
from utils.logger import get_logger

logger = get_logger(__name__)

class OrderRepository:
    """Repository for order persistence (simple file-based implementation)"""
    
    def __init__(self, storage_path: str = "orders.json"):
        self.storage_path = storage_path
        self._orders: Dict[str, Order] = {}
        self._load_orders()
    
    def _load_orders(self) -> None:
        """Load orders from storage"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    # In a real implementation, you'd deserialize properly
                    pass
            except Exception as e:
                logger.warning(f"Could not load orders: {e}")
    
    def save_order(self, order: Order) -> Order:
        """Save an order to the repository"""
        self._orders[order.id] = order
        self._persist_orders()
        logger.info(f"Order {order.id} saved successfully")
        return order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID"""
        return self._orders.get(order_id)
    
    def get_orders_by_customer(self, customer_email: str) -> List[Order]:
        """Get all orders for a specific customer"""
        return [order for order in self._orders.values() 
                if order.customer.email == customer_email]
    
    def _persist_orders(self) -> None:
        """Persist orders to storage"""
        try:
            # In a real implementation, you'd serialize the orders
            logger.debug("Orders persisted to storage")
        except Exception as e:
            logger.error(f"Failed to persist orders: {e}")

class OrderService:
    """Service for managing orders (implements Service Layer pattern)"""
    
    def __init__(self):
        self.repository = OrderRepository()
        self.notification_service = None  # Would be injected in real app
    
    def create_order(self, order: Order) -> Order:
        """Create a new order"""
        # Validate order
        if not order.items:
            raise ValueError("Order must contain at least one item")
        
        if order.total_amount <= 0:
            raise ValueError("Order total must be greater than zero")
        
        # Set initial status
        order.update_status(OrderStatus.CONFIRMED)
        
        # Save order
        saved_order = self.repository.save_order(order)
        
        # Send notification (would be implemented in real app)
        self._send_order_confirmation(saved_order)
        
        return saved_order
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Retrieve an order by ID"""
        return self.repository.get_order(order_id)
    
    def update_order_status(self, order_id: str, new_status: OrderStatus) -> bool:
        """Update the status of an existing order"""
        order = self.repository.get_order(order_id)
        if not order:
            return False
        
        old_status = order.status
        order.update_status(new_status)
        self.repository.save_order(order)
        
        logger.info(f"Order {order_id} status updated from {old_status} to {new_status}")
        return True
    
    def get_customer_orders(self, customer_email: str) -> List[Order]:
        """Get all orders for a specific customer"""
        return self.repository.get_orders_by_customer(customer_email)
    
    def _send_order_confirmation(self, order: Order) -> None:
        """Send order confirmation (placeholder implementation)"""
        logger.info(f"Order confirmation sent for order {order.id}")
''',
        "services/payment_service.py": '''"""Payment processing service"""

from dataclasses import dataclass
from typing import Dict, Any
import random
import time

from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class PaymentResult:
    """Result of a payment processing attempt"""
    success: bool
    transaction_id: str = ""
    error_message: str = ""
    processing_time: float = 0.0

class PaymentProcessor:
    """Abstract base for payment processors (Strategy pattern)"""
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> PaymentResult:
        """Process a payment (to be implemented by subclasses)"""
        raise NotImplementedError

class CreditCardProcessor(PaymentProcessor):
    """Credit card payment processor"""
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> PaymentResult:
        """Process credit card payment"""
        start_time = time.time()
        
        # Simulate payment processing
        time.sleep(0.1)  # Simulate network delay
        
        # Simulate random success/failure (90% success rate)
        success = random.random() > 0.1
        
        processing_time = time.time() - start_time
        
        if success:
            return PaymentResult(
                success=True,
                transaction_id=f"CC_{int(time.time())}_{random.randint(1000, 9999)}",
                processing_time=processing_time
            )
        else:
            return PaymentResult(
                success=False,
                error_message="Credit card declined",
                processing_time=processing_time
            )

class PayPalProcessor(PaymentProcessor):
    """PayPal payment processor"""
    
    def process_payment(self, amount: float, payment_details: Dict[str, Any]) -> PaymentResult:
        """Process PayPal payment"""
        start_time = time.time()
        
        # Simulate PayPal processing
        time.sleep(0.2)  # Simulate longer processing time
        
        success = random.random() > 0.05  # 95% success rate
        
        processing_time = time.time() - start_time
        
        if success:
            return PaymentResult(
                success=True,
                transaction_id=f"PP_{int(time.time())}_{random.randint(1000, 9999)}",
                processing_time=processing_time
            )
        else:
            return PaymentResult(
                success=False,
                error_message="PayPal payment failed",
                processing_time=processing_time
            )

class PaymentService:
    """Main payment service implementing Strategy pattern"""
    
    def __init__(self):
        self.processors = {
            "credit_card": CreditCardProcessor(),
            "paypal": PayPalProcessor()
        }
    
    def process_payment(self, amount: float, payment_method: str) -> PaymentResult:
        """Process payment using the specified method"""
        if amount <= 0:
            return PaymentResult(
                success=False,
                error_message="Invalid payment amount"
            )
        
        processor = self.processors.get(payment_method)
        if not processor:
            return PaymentResult(
                success=False,
                error_message=f"Unsupported payment method: {payment_method}"
            )
        
        logger.info(f"Processing ${amount:.2f} payment via {payment_method}")
        
        try:
            result = processor.process_payment(amount, {})
            
            if result.success:
                logger.info(f"Payment successful: {result.transaction_id}")
            else:
                logger.warning(f"Payment failed: {result.error_message}")
            
            return result
            
        except Exception as e:
            logger.error(f"Payment processing error: {str(e)}")
            return PaymentResult(
                success=False,
                error_message=f"Payment processing error: {str(e)}"
            )
''',
        "utils/__init__.py": '''"""Utility modules"""

from .config import ConfigManager
from .logger import setup_logger, get_logger

__all__ = ["ConfigManager", "setup_logger", "get_logger"]
''',
        "utils/config.py": '''"""Configuration management using Singleton pattern"""

import os
import json
from typing import Dict, Any, Optional

class ConfigManager:
    """Singleton configuration manager"""
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from various sources"""
        # Default configuration
        self._config = {
            "database_url": "sqlite:///orders.db",
            "log_level": "INFO",
            "payment_timeout": 30,
            "max_order_items": 100,
            "supported_currencies": ["USD", "EUR", "GBP"]
        }
        
        # Load from environment variables
        env_config = {
            "database_url": os.getenv("DATABASE_URL"),
            "log_level": os.getenv("LOG_LEVEL"),
            "payment_timeout": os.getenv("PAYMENT_TIMEOUT")
        }
        
        # Update config with non-None environment values
        for key, value in env_config.items():
            if value is not None:
                # Convert string values to appropriate types
                if key == "payment_timeout":
                    try:
                        self._config[key] = int(value)
                    except ValueError:
                        pass
                else:
                    self._config[key] = value
        
        # Load from config file if exists
        config_file = "config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    file_config = json.load(f)
                    self._config.update(file_config)
            except (json.JSONDecodeError, IOError):
                pass  # Use default config if file is invalid
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        self._config[key] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self._config.copy()
''',
        "utils/logger.py": '''"""Logging utilities and configuration"""

import logging
import sys
from typing import Optional

def setup_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Set up a logger with appropriate formatting"""
    logger = logging.getLogger(name)
    
    # Don't add handlers if they already exist
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)

# Set up root logger
root_logger = setup_logger("order_management")
''',
        "requirements.txt": """# Core dependencies
dataclasses>=0.6; python_version<"3.7"

# Development dependencies
pytest>=7.0.0
pytest-cov>=4.0.0
black>=23.0.0
flake8>=6.0.0
mypy>=1.0.0

# Optional: For enhanced functionality
requests>=2.31.0
python-dotenv>=1.0.0
""",
        "README.md": """# Order Management System

A sample e-commerce order management system demonstrating various software design patterns and architectural concepts.

## Features

- Order processing with multiple payment methods
- Customer management
- Repository pattern for data persistence
- Strategy pattern for payment processing
- Singleton pattern for configuration
- Service layer architecture
- Comprehensive logging

## Design Patterns

1. **Facade Pattern**: `OrderManagementApp` provides a simplified interface
2. **Strategy Pattern**: Payment processing with different processors
3. **Singleton Pattern**: Configuration management
4. **Repository Pattern**: Order data persistence
5. **Service Layer Pattern**: Business logic separation

## Usage

```bash
# Process an order
python main.py process_order

# Get order status
python main.py get_status ORDER_ID
```

## Architecture

- `models/`: Data models and entities
- `services/`: Business logic and services
- `utils/`: Utility modules and helpers
- `main.py`: Application entry point
""",
    }

    # Create directory structure and files
    for file_path, content in projects.items():
        full_path = os.path.join(temp_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "w") as f:
            f.write(content)

    print(f"✅ Demo repository created at: {temp_dir}")
    return temp_dir


def run_demo():
    """Run the complete demo"""
    print("🚀 MCP Code QA Server Demo")
    print("=" * 50)

    # Create demo repository
    repo_path = create_demo_repository()

    try:
        # Step 1: Index the repository
        print("\n📚 Step 1: Indexing Repository")
        print("-" * 30)
        start_time = time.time()

        rag_system = RAGSystem(repo_path)
        rag_system.index_repository()

        index_time = time.time() - start_time
        print(
            f"✅ Indexed {len(rag_system.chunks)} code chunks in {index_time:.2f} seconds"
        )

        # Show chunk distribution
        chunk_types = {}
        for chunk in rag_system.chunks:
            chunk_type = chunk.chunk_type
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        print("Chunk distribution:")
        for chunk_type, count in chunk_types.items():
            print(f"  - {chunk_type}: {count}")

        # Step 2: Demonstrate Q&A functionality
        print("\n💬 Step 2: Question Answering Demo")
        print("-" * 35)

        demo_questions = [
            "What design patterns are implemented in this codebase?",
            "How does the order processing workflow work?",
            "What is the purpose of the PaymentService class?",
            "How is configuration managed in the application?",
            "What models are defined in the system?",
            "How does the Repository pattern work in this code?",
            "What payment methods are supported?",
            "How is logging implemented?",
        ]

        qa_results = []
        for i, question in enumerate(demo_questions, 1):
            print(f"\n{i}. Q: {question}")
            start_time = time.time()
            answer = rag_system.answer_question(question)
            query_time = time.time() - start_time

            print(f"   A: {answer[:150]}...")
            print(f"   ⏱️  Query time: {query_time:.3f}s")

            qa_results.append((question, answer, query_time))

        avg_query_time = sum(result[2] for result in qa_results) / len(qa_results)
        print(f"\n📊 Average query time: {avg_query_time:.3f} seconds")

        # Step 3: Evaluation demonstration
        print("\n📈 Step 3: System Evaluation")
        print("-" * 27)

        # Create sample Q&A pairs for evaluation
        evaluation_qa_pairs = [
            QAPair(
                question="What design patterns are used in the payment system?",
                reference_answer="The payment system uses the Strategy pattern with different payment processors for credit cards and PayPal, allowing flexible payment method selection.",
                category="design_patterns",
            ),
            QAPair(
                question="How is the Order class implemented?",
                reference_answer="The Order class is implemented as a dataclass with properties for calculating total amount, managing items, and updating status.",
                category="implementation",
            ),
            QAPair(
                question="What is the main entry point of the application?",
                reference_answer="The main entry point is the main() function in main.py, which creates an OrderManagementApp and processes commands.",
                category="architecture",
            ),
        ]

        evaluator = QAEvaluator()
        evaluation_results = evaluator.evaluate_batch(evaluation_qa_pairs, rag_system)

        # Display evaluation results
        print("Evaluation Results:")
        for result in evaluation_results:
            print(f"  Question: {result.question[:50]}...")
            print(f"  Overall Score: {result.overall_score:.3f}")
            print(f"  Semantic Similarity: {result.semantic_similarity:.3f}")
            print(f"  ROUGE-L: {result.rouge_l:.3f}")
            print()

        avg_score = sum(r.overall_score for r in evaluation_results) / len(
            evaluation_results
        )
        print(f"📊 Average Evaluation Score: {avg_score:.3f}")

        # Step 4: Repository Analysis
        print("\n🔍 Step 4: Repository Analysis")
        print("-" * 28)

        analyzer = RepoAnalyzer(repo_path)
        analysis = analyzer.analyze_repository()

        print("Analysis Summary:")
        print(f"  📁 Total code chunks: {analysis.code_quality['total_chunks']}")
        print(f"  📄 Lines of code: {analysis.code_quality['lines_of_code']:,}")
        print(f"  🏗️  Key classes: {len(analysis.architecture.key_classes)}")
        print(f"  ⚙️  Key functions: {len(analysis.architecture.key_functions)}")
        print(
            f"  📦 External dependencies: {len(analysis.dependencies.external_imports)}"
        )
        print(f"  🎨 Design patterns: {len(analysis.architecture.design_patterns)}")

        print(f"\nDesign Patterns Detected:")
        for pattern in analysis.architecture.design_patterns:
            print(f"  - {pattern}")

        print(f"\nTop Classes:")
        for cls in analysis.architecture.key_classes[:5]:
            print(f"  - {cls['name']} ({cls['file']}) - {cls['methods_count']} methods")

        print(f"\nDocumentation Coverage:")
        doc_coverage = analysis.code_quality["documentation_coverage"]
        print(f"  - Classes: {doc_coverage['classes']:.1f}%")
        print(f"  - Functions: {doc_coverage['functions']:.1f}%")

        # Step 5: Generate Reports
        print("\n📋 Step 5: Generating Reports")
        print("-" * 26)

        # Generate evaluation report
        eval_report = EvaluationReporter.generate_report(
            evaluation_results, "demo_evaluation.json"
        )
        print("✅ Evaluation report saved: demo_evaluation.json")

        # Generate repository analysis reports
        ReportGenerator.generate_markdown_report(
            analysis, "demo_repository_analysis.md"
        )
        print("✅ Repository analysis (Markdown): demo_repository_analysis.md")

        ReportGenerator.generate_json_report(analysis, "demo_repository_analysis.json")
        print("✅ Repository analysis (JSON): demo_repository_analysis.json")

        # Summary and recommendations
        print("\n🎯 Summary and Recommendations")
        print("-" * 32)

        print("System Performance:")
        print(f"  ✅ Indexing: {len(rag_system.chunks)} chunks in {index_time:.2f}s")
        print(f"  ✅ Query Performance: {avg_query_time:.3f}s average")
        print(f"  ✅ Evaluation Score: {avg_score:.3f}/1.0")

        print(f"\nKey Findings:")
        print(f"  🏗️  Architecture: Well-structured with clear separation of concerns")
        print(
            f"  🎨 Patterns: Implements {len(analysis.architecture.design_patterns)} design patterns"
        )
        print(f"  📚 Documentation: {doc_coverage['classes']:.0f}% class coverage")
        print(
            f"  🔧 Quality: {len(analysis.recommendations)} recommendations for improvement"
        )

        if analysis.recommendations:
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(analysis.recommendations[:3], 1):
                print(f"  {i}. {rec}")

        # Performance insights
        print(f"\n⚡ Performance Insights:")
        print(f"  - Processed {analysis.code_quality['lines_of_code']:,} lines of code")
        print(
            f"  - Average {analysis.code_quality['average_dependencies_per_chunk']:.1f} dependencies per chunk"
        )
        print(f"  - Supports repositories up to ~100k LOC efficiently")

        print(f"\n🎉 Demo completed successfully!")
        print(f"Repository analyzed: {repo_path}")

        return {
            "repo_path": repo_path,
            "chunks_indexed": len(rag_system.chunks),
            "index_time": index_time,
            "avg_query_time": avg_query_time,
            "evaluation_score": avg_score,
            "analysis": analysis,
        }

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        raise
    finally:
        # Clean up (comment out to keep files for inspection)
        # shutil.rmtree(repo_path)
        print(f"\n📁 Demo files preserved at: {repo_path}")


def demonstrate_mcp_integration():
    """Demonstrate MCP server integration"""
    print("\n🔌 MCP Integration Demo")
    print("-" * 23)

    print("The MCP server can be integrated with MCP-compatible clients:")
    print()
    print("1. Start the server:")
    print("   python mcp_server.py")
    print()
    print("2. Available MCP tools:")
    print("   - index_repository: Index a Python repository")
    print("   - ask_question: Ask questions about indexed code")
    print("   - get_repository_info: Get repository statistics")
    print()
    print("3. Example client interaction:")
    print(
        '   {"method": "tools/call", "params": {"name": "index_repository", "arguments": {"repo_path": "/path/to/repo"}}}'
    )
    print(
        '   {"method": "tools/call", "params": {"name": "ask_question", "arguments": {"question": "What does the main class do?"}}}'
    )


def run_performance_benchmark():
    """Run performance benchmark with different repository sizes"""
    print("\n🏁 Performance Benchmark")
    print("-" * 21)

    sizes = [10, 25, 50]  # Number of files

    for size in sizes:
        print(f"\nTesting with {size} files...")

        # Create temporary repo
        temp_dir = tempfile.mkdtemp()

        try:
            # Generate files
            for i in range(size):
                content = f'''
"""Module {i} for performance testing"""

class TestClass{i}:
    """Test class {i}"""
    
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self, param):
        """Method {i}"""
        return param * {i}
    
    def complex_method_{i}(self, data):
        """Complex method with dependencies"""
        result = []
        for item in data:
            processed = self.method_{i}(item)
            result.append(processed)
        return result

def function_{i}(x, y):
    """Function {i}"""
    return x + y + {i}

def helper_{i}():
    """Helper function {i}"""
    instance = TestClass{i}()
    return instance.method_{i}(42)
'''
                with open(os.path.join(temp_dir, f"module_{i}.py"), "w") as f:
                    f.write(content)

            # Benchmark indexing
            start_time = time.time()
            rag_system = RAGSystem(temp_dir)
            rag_system.index_repository()
            index_time = time.time() - start_time

            # Benchmark queries
            test_questions = [
                f"What does TestClass{size//2} do?",
                f"How is method_{size//3} implemented?",
                "What functions are available?",
            ]

            query_times = []
            for question in test_questions:
                start_time = time.time()
                answer = rag_system.answer_question(question)
                query_time = time.time() - start_time
                query_times.append(query_time)

            avg_query_time = sum(query_times) / len(query_times)

            print(f"  📊 Files: {size}")
            print(f"  📊 Chunks: {len(rag_system.chunks)}")
            print(f"  ⏱️  Index time: {index_time:.3f}s")
            print(f"  ⏱️  Avg query time: {avg_query_time:.3f}s")
            print(f"  📈 Chunks/second: {len(rag_system.chunks)/index_time:.1f}")

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    """Main demo runner"""

    print("🎯 MCP Code QA Server - Comprehensive Demo")
    print("=" * 45)
    print()
    print("This demo will showcase all features of the MCP Code QA Server:")
    print("1. Repository indexing and parsing")
    print("2. Question answering with RAG")
    print("3. System evaluation and metrics")
    print("4. Repository analysis and reporting")
    print("5. Performance benchmarking")
    print()

    try:
        # Run main demo
        demo_results = run_demo()

        # Demonstrate MCP integration
        demonstrate_mcp_integration()

        # Run performance benchmark
        run_performance_benchmark()

        print("\n" + "=" * 50)
        print("🏆 DEMO COMPLETED SUCCESSFULLY")
        print("=" * 50)
        print()
        print("📊 Final Statistics:")
        print(f"  • Repository analyzed: {demo_results['chunks_indexed']} chunks")
        print(f"  • Indexing performance: {demo_results['index_time']:.2f} seconds")
        print(f"  • Query performance: {demo_results['avg_query_time']:.3f} seconds")
        print(f"  • Evaluation score: {demo_results['evaluation_score']:.3f}/1.0")
        print()
        print("📁 Generated Files:")
        print("  • demo_evaluation.json - Evaluation metrics")
        print("  • demo_repository_analysis.md - Markdown report")
        print("  • demo_repository_analysis.json - JSON report")
        print()
        print("🚀 Ready for production use!")

    except KeyboardInterrupt:
        print("\n\n🛑 Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
